//! Small recursive building blocks shared by the amp stages.
//!
//! Every struct here keeps its state flushed through
//! [`denormal::flush`][crate::dsp::denormal::flush] so that decaying tails
//! cannot stall the FPU on targets where the MXCSR guard is unavailable
//! (`CLAUDE.md` §1).

use super::denormal::flush;
use super::{one_pole_coeff, time_constant_coeff};

/// First-order high-pass used as the inter-stage DC blocker.
///
/// Specification section 2.A: "A 1st-order high-pass filter (f_c = 10 Hz)
/// follows each gain stage to remove DC bias accumulation." Physically this is
/// the coupling capacitor into the next grid leak resistor; the Koren plate
/// model produces a large standing DC plate voltage that must not reach the
/// next stage's operating point.
#[derive(Debug, Clone, Copy, Default)]
pub struct DcBlocker {
    /// One-pole lowpass coefficient tracking the DC component.
    coeff: f32,
    /// Running estimate of the DC offset.
    state: f32,
}

impl DcBlocker {
    /// Nominal corner frequency in Hz, per specification section 2.A.
    pub const CUTOFF_HZ: f32 = 10.0;

    /// Prepares the blocker for `sample_rate` (the *oversampled* rate) and
    /// clears its state.
    pub fn prepare(&mut self, sample_rate: f32) {
        self.coeff = one_pole_coeff(Self::CUTOFF_HZ, sample_rate);
        self.reset();
    }

    /// Clears the DC estimate.
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Removes the tracked DC component from `x`.
    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        self.state = flush(self.state + self.coeff * (x - self.state));
        x - self.state
    }
}

/// First-order lowpass, used for cathode-bypass networks, Miller capacitance
/// rolloff and the transformer's leakage-inductance pole.
#[derive(Debug, Clone, Copy, Default)]
pub struct OnePoleLp {
    coeff: f32,
    state: f32,
}

impl OnePoleLp {
    /// Sets the corner frequency and clears the state.
    pub fn prepare(&mut self, cutoff_hz: f32, sample_rate: f32) {
        self.coeff = one_pole_coeff(cutoff_hz, sample_rate);
        self.reset();
    }

    /// Changes the corner frequency while preserving the current state.
    pub fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        self.coeff = one_pole_coeff(cutoff_hz, sample_rate);
    }

    /// Clears the filter state.
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Forces the filter state, used to seed a stage at its quiescent operating
    /// point so the plugin does not thump on the first block.
    pub fn preload(&mut self, value: f32) {
        self.state = value;
    }

    /// Current filter state, i.e. the lowpassed value.
    #[inline(always)]
    pub fn value(&self) -> f32 {
        self.state
    }

    /// Filters one sample.
    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        self.state = flush(self.state + self.coeff * (x - self.state));
        self.state
    }
}

/// Dual-time-constant peak follower.
///
/// Specification section 2.C requires the B+ sag tracker to use an 8 ms attack
/// and a 120 ms release. The same structure also drives the jewel-lamp
/// brightness in [`engine`][crate::dsp::engine].
#[derive(Debug, Clone, Copy, Default)]
pub struct EnvelopeFollower {
    attack: f32,
    release: f32,
    state: f32,
}

impl EnvelopeFollower {
    /// Configures the two time constants in milliseconds.
    pub fn prepare(&mut self, attack_ms: f32, release_ms: f32, sample_rate: f32) {
        self.attack = time_constant_coeff(attack_ms, sample_rate);
        self.release = time_constant_coeff(release_ms, sample_rate);
        self.reset();
    }

    /// Clears the envelope.
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Current envelope value.
    #[inline(always)]
    pub fn value(&self) -> f32 {
        self.state
    }

    /// Feeds one rectified sample into the follower.
    #[inline(always)]
    pub fn process(&mut self, rectified: f32) -> f32 {
        let coeff = if rectified > self.state {
            self.attack
        } else {
            self.release
        };
        self.state = flush(self.state + coeff * (rectified - self.state));
        self.state
    }
}

/// Linear parameter ramp for switches that would otherwise click.
///
/// `BoolParam` carries no smoother of its own, but specification section 3
/// asks for 10 ms smoothing on `global_boost`, and hard-switching the gain
/// boost stage or the channel would produce a step discontinuity. This ramp
/// provides a bounded, allocation-free crossfade weight in `0.0..=1.0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SwitchRamp {
    step: f32,
    current: f32,
    target: f32,
}

impl SwitchRamp {
    /// Configures the ramp duration and snaps to `initial`.
    pub fn prepare(&mut self, ramp_ms: f32, sample_rate: f32, initial: bool) {
        let samples = (ramp_ms * 0.001 * sample_rate).max(1.0);
        self.step = 1.0 / samples;
        self.current = if initial { 1.0 } else { 0.0 };
        self.target = self.current;
    }

    /// Requests a new switch position.
    #[inline(always)]
    pub fn set(&mut self, engaged: bool) {
        self.target = if engaged { 1.0 } else { 0.0 };
    }

    /// Advances the ramp by one sample and returns the crossfade weight.
    ///
    /// Deliberately not named `next`: this is not an iterator, and a method
    /// with that name on a non-`Iterator` type reads as one at the call site.
    #[inline(always)]
    pub fn advance(&mut self) -> f32 {
        if self.current < self.target {
            self.current = (self.current + self.step).min(self.target);
        } else if self.current > self.target {
            self.current = (self.current - self.step).max(self.target);
        }
        self.current
    }

    /// Current weight without advancing.
    #[inline(always)]
    pub fn value(&self) -> f32 {
        self.current
    }

    /// True when the ramp has fully settled at `0.0`.
    #[inline(always)]
    pub fn is_fully_off(&self) -> bool {
        self.current <= 0.0 && self.target <= 0.0
    }
}

/// Transposed-direct-form-II biquad.
///
/// Used for the fixed voicing filters (bright cap, presence shelf, speaker
/// resonance) where a full circuit network would be overkill. TDF-II is chosen
/// over DF-I because it needs only two state words and has better
/// coefficient-quantization behaviour at the low corner frequencies used by the
/// 85 Hz cabinet resonance.
#[derive(Debug, Clone, Copy, Default)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl Biquad {
    /// Builds a biquad from already-normalized (a0 == 1) coefficients.
    pub fn from_coefficients(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            s1: 0.0,
            s2: 0.0,
        }
    }

    /// Second-order high-pass via the RBJ cookbook bilinear design.
    pub fn highpass(cutoff_hz: f32, q: f32, sample_rate: f32) -> Self {
        let (alpha, cos_w0) = Self::rbj_common(cutoff_hz, q, sample_rate);
        let a0 = 1.0 + alpha;
        let b0 = (1.0 + cos_w0) * 0.5;
        let b1 = -(1.0 + cos_w0);
        let b2 = b0;
        Self::from_coefficients(
            b0 / a0,
            b1 / a0,
            b2 / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha) / a0,
        )
    }

    /// Second-order low-pass via the RBJ cookbook bilinear design.
    pub fn lowpass(cutoff_hz: f32, q: f32, sample_rate: f32) -> Self {
        let (alpha, cos_w0) = Self::rbj_common(cutoff_hz, q, sample_rate);
        let a0 = 1.0 + alpha;
        let b1 = 1.0 - cos_w0;
        let b0 = b1 * 0.5;
        Self::from_coefficients(
            b0 / a0,
            b1 / a0,
            b0 / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha) / a0,
        )
    }

    /// Constant-skirt-gain peaking EQ via the RBJ cookbook bilinear design.
    pub fn peaking(cutoff_hz: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        let a = 10.0f32.powf(gain_db / 40.0);
        let (alpha, cos_w0) = Self::rbj_common(cutoff_hz, q, sample_rate);
        let a0 = 1.0 + alpha / a;
        Self::from_coefficients(
            (1.0 + alpha * a) / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha * a) / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha / a) / a0,
        )
    }

    /// Constant-slope low shelf via the RBJ cookbook bilinear design.
    ///
    /// Approaches `gain_db` below `cutoff_hz` and unity above it. A guitar
    /// speaker's low-mid weight is a shelf, not a resonance: the measured
    /// response of a 4x12 sits several dB above its 1 kHz level from the
    /// cabinet's tuning frequency all the way up to the midrange, and no
    /// combination of peaking sections reproduces that plateau.
    pub fn low_shelf(cutoff_hz: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        let a = 10.0f32.powf(gain_db / 40.0);
        let (alpha, cos_w0) = Self::rbj_common(cutoff_hz, q, sample_rate);
        let shelf_alpha = 2.0 * a.sqrt() * alpha;
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + shelf_alpha;
        Self::from_coefficients(
            a * ((a + 1.0) - (a - 1.0) * cos_w0 + shelf_alpha) / a0,
            2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0) / a0,
            a * ((a + 1.0) - (a - 1.0) * cos_w0 - shelf_alpha) / a0,
            -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0) / a0,
            ((a + 1.0) + (a - 1.0) * cos_w0 - shelf_alpha) / a0,
        )
    }

    /// Shared `alpha` and `cos(w0)` terms of the RBJ designs. The cutoff is
    /// clamped to 45 % of Nyquist so a badly-behaved caller cannot produce an
    /// unstable pole pair.
    fn rbj_common(cutoff_hz: f32, q: f32, sample_rate: f32) -> (f32, f32) {
        let nyquist = sample_rate * 0.5;
        let f = cutoff_hz.clamp(1.0, nyquist * 0.9);
        let w0 = std::f32::consts::TAU * f / sample_rate;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q.max(0.05));
        (alpha, cos_w0)
    }

    /// Clears the filter state.
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    /// Filters one sample.
    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.s1;
        self.s1 = flush(self.b1 * x - self.a1 * y + self.s2);
        self.s2 = flush(self.b2 * x - self.a2 * y);
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f32 = 48_000.0;

    #[test]
    fn dc_blocker_removes_constant_offset() {
        let mut blocker = DcBlocker::default();
        blocker.prepare(FS);
        let mut last = 0.0;
        for _ in 0..(FS as usize) {
            last = blocker.process(1.0);
        }
        assert!(last.abs() < 1.0e-3, "residual DC {last}");
    }

    #[test]
    fn dc_blocker_passes_audio_band() {
        let mut blocker = DcBlocker::default();
        blocker.prepare(FS);
        let mut peak: f32 = 0.0;
        for n in 0..(FS as usize) {
            let x = (std::f32::consts::TAU * 440.0 * n as f32 / FS).sin();
            let y = blocker.process(x);
            if n > FS as usize / 2 {
                peak = peak.max(y.abs());
            }
        }
        assert!(peak > 0.99, "440 Hz was attenuated to {peak}");
    }

    #[test]
    fn envelope_attack_is_faster_than_release() {
        let mut env = EnvelopeFollower::default();
        env.prepare(8.0, 120.0, FS);
        for _ in 0..(0.05 * FS) as usize {
            env.process(1.0);
        }
        let charged = env.value();
        assert!(charged > 0.98, "attack only reached {charged}");

        for _ in 0..(0.05 * FS) as usize {
            env.process(0.0);
        }
        // 50 ms into a 120 ms release the envelope should still hold >55 %.
        assert!(
            env.value() > 0.55,
            "release decayed too fast: {}",
            env.value()
        );
    }

    #[test]
    fn switch_ramp_is_bounded_and_monotonic() {
        let mut ramp = SwitchRamp::default();
        ramp.prepare(10.0, FS, false);
        ramp.set(true);
        let mut previous = 0.0;
        for _ in 0..(0.02 * FS) as usize {
            let v = ramp.advance();
            assert!((0.0..=1.0).contains(&v));
            assert!(v >= previous);
            previous = v;
        }
        assert_eq!(previous, 1.0);
    }

    #[test]
    fn biquad_designs_are_stable() {
        // |a2| < 1 and |a1| < 1 + a2 is the standard stability triangle.
        for filter in [
            Biquad::highpass(85.0, 1.2, FS),
            Biquad::lowpass(4_800.0, 0.707, FS),
            Biquad::peaking(2_200.0, 2.0, 6.0, FS),
            Biquad::low_shelf(700.0, 1.07, 13.6, FS),
        ] {
            assert!(filter.a2.abs() < 1.0);
            assert!(filter.a1.abs() < 1.0 + filter.a2);
        }
    }

    #[test]
    fn low_shelf_lifts_the_bottom_and_leaves_the_top_alone() {
        // Quadrature correlation rather than peak-of-samples: at 8 kHz a
        // 48 kHz sine has six samples per cycle, and the largest *sample* can
        // sit a decibel below the true peak.
        let gain_at = |frequency: f32| -> f32 {
            let mut filter = Biquad::low_shelf(700.0, 1.07, 13.6, FS);
            let settle = (FS / frequency * 20.0) as usize;
            let measure = (FS / frequency * 40.0) as usize;
            for n in 0..settle {
                filter.process((std::f32::consts::TAU * frequency * n as f32 / FS).sin());
            }
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for n in 0..measure {
                let phase = std::f32::consts::TAU * frequency * (settle + n) as f32 / FS;
                let y = filter.process(phase.sin()) as f64;
                let angle = std::f64::consts::TAU * frequency as f64 * n as f64 / FS as f64;
                re += y * angle.cos();
                im += y * angle.sin();
            }
            let magnitude = 2.0 * (re * re + im * im).sqrt() / measure as f64;
            20.0 * magnitude.log10() as f32
        };

        // Deep in the shelf the gain is the full 13.6 dB; well above the
        // corner the filter is transparent; the corner itself sits near half.
        let low = gain_at(60.0);
        let high = gain_at(8_000.0);
        let corner = gain_at(700.0);
        assert!((low - 13.6).abs() < 0.6, "shelf floor was {low} dB");
        assert!(high.abs() < 0.3, "shelf coloured 8 kHz by {high} dB");
        assert!(
            (2.0..=12.0).contains(&corner),
            "corner gain {corner} dB is not between the two plateaus"
        );
    }

    #[test]
    fn biquad_lowpass_attenuates_above_cutoff() {
        let mut filter = Biquad::lowpass(1_000.0, 0.707, FS);
        let mut peak: f32 = 0.0;
        for n in 0..4_800 {
            let x = (std::f32::consts::TAU * 10_000.0 * n as f32 / FS).sin();
            let y = filter.process(x);
            if n > 2_400 {
                peak = peak.max(y.abs());
            }
        }
        // A 2nd-order pole pair a decade up is -40 dB.
        assert!(peak < 0.02, "10 kHz leaked through at {peak}");
    }
}
