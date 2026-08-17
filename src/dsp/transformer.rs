//! Output transformer core model.
//!
//! Specification section 2 places an "Output Transformer Core" block between
//! the 8x downsampler and the cabinet convolution. A guitar amplifier's output
//! transformer is not a wire: it band-limits at both ends and, crucially, its
//! core saturates on low-frequency transients, which is a large part of why a
//! cranked amp thickens rather than simply clipping.
//!
//! # Model
//!
//! The primary winding is a shunt inductance `Lp` fed from the tubes' plate
//! resistance `Rp`. Magnetizing current is the integral of the applied voltage,
//! so with `phi` the (normalized) core flux:
//!
//! ```text
//! phi        = lowpass(v, fL),          fL = Rp / (2*pi*Lp)
//! i_mag      = phi + beta * phi^3       (saturating core reluctance)
//! v_secondary = v - i_mag
//! ```
//!
//! The linear part `v - phi` is exactly a first-order high-pass at `fL`, the
//! transformer's low-frequency corner. The cubic term only becomes significant
//! once the flux approaches the core's saturation knee, producing the
//! odd-harmonic low-end compression a real core exhibits. A 20 H primary loaded
//! by an EL34 pair's ~2 kΩ plate resistance puts `fL` at
//! `2000 / (2*pi*20) ~= 16 Hz`.
//!
//! Leakage inductance resonating against the winding capacitance gives the
//! high-frequency corner, modelled as a lightly resonant second-order lowpass
//! at 11 kHz with `Q = 1.1` — the small peak before the rolloff is the
//! transformer's own presence bump.

use super::denormal::sanitize;
use super::filters::{Biquad, OnePoleLp};

/// Low-frequency corner set by the primary inductance and plate resistance.
const PRIMARY_CORNER_HZ: f32 = 16.0;
/// Leakage-inductance/winding-capacitance resonance.
const LEAKAGE_RESONANCE_HZ: f32 = 11_000.0;
/// Quality factor of that resonance; slightly above critical, giving ~1 dB of
/// lift before the rolloff.
const LEAKAGE_Q: f32 = 1.1;
/// Cubic coefficient of the core's reluctance curve.
///
/// Chosen so the magnetizing current departs from linear by about 30 % once the
/// normalized flux reaches 0.7, which is where a correctly sized guitar-amp
/// output transformer begins to saturate.
const CORE_SATURATION_BETA: f32 = 0.6;
/// Hard bound on the flux used in the cubic term, so a pathological input
/// cannot produce an unbounded `phi^3` (`CLAUDE.md` §1).
const FLUX_LIMIT: f32 = 4.0;

/// Output transformer, running at the host sample rate.
#[repr(align(64))]
#[derive(Debug, Clone, Default)]
pub struct OutputTransformer {
    /// Core flux, as a lowpass of the applied primary voltage.
    flux: OnePoleLp,
    /// Leakage-inductance rolloff.
    leakage: Option<Biquad>,
}

impl OutputTransformer {
    /// Builds an unprepared transformer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures both corners for the host `sample_rate`.
    pub fn prepare(&mut self, sample_rate: f32) {
        self.flux.prepare(PRIMARY_CORNER_HZ, sample_rate);
        self.leakage = Some(Biquad::lowpass(
            LEAKAGE_RESONANCE_HZ,
            LEAKAGE_Q,
            sample_rate,
        ));
    }

    /// Clears the core flux and the leakage filter state.
    pub fn reset(&mut self) {
        self.flux.reset();
        if let Some(leakage) = self.leakage.as_mut() {
            leakage.reset();
        }
    }

    /// Processes one sample at the host rate.
    #[inline(always)]
    pub fn process(&mut self, input: f32) -> f32 {
        let flux = self.flux.process(input).clamp(-FLUX_LIMIT, FLUX_LIMIT);
        let magnetizing = flux + CORE_SATURATION_BETA * flux * flux * flux;
        let secondary = input - magnetizing;
        let output = match self.leakage.as_mut() {
            Some(leakage) => leakage.process(secondary),
            // Unprepared: pass the core model through unfiltered rather than
            // silencing the amp.
            None => secondary,
        };
        sanitize(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f32 = 48_000.0;

    fn prepared() -> OutputTransformer {
        let mut transformer = OutputTransformer::new();
        transformer.prepare(FS);
        transformer
    }

    fn peak_response(transformer: &mut OutputTransformer, frequency: f32, amplitude: f32) -> f32 {
        transformer.reset();
        let cycles = 24.0;
        let total = (FS / frequency * cycles) as usize;
        let mut peak = 0.0f32;
        for n in 0..total {
            let x = amplitude * (std::f32::consts::TAU * frequency * n as f32 / FS).sin();
            let y = transformer.process(x);
            if n > total / 2 {
                peak = peak.max(y.abs());
            }
        }
        peak
    }

    #[test]
    fn midband_passes_at_unity() {
        let mut transformer = prepared();
        let gain = peak_response(&mut transformer, 1_000.0, 0.1) / 0.1;
        assert!((gain - 1.0).abs() < 0.1, "1 kHz gain was {gain}");
    }

    #[test]
    fn low_end_rolls_off_below_the_primary_corner() {
        let mut transformer = prepared();
        let low = peak_response(&mut transformer, 5.0, 0.1) / 0.1;
        let mid = peak_response(&mut transformer, 1_000.0, 0.1) / 0.1;
        assert!(low < mid * 0.4, "5 Hz was only attenuated to {low}");
    }

    #[test]
    fn high_end_rolls_off_above_the_leakage_resonance() {
        let mut transformer = prepared();
        let mid = peak_response(&mut transformer, 1_000.0, 0.1) / 0.1;
        let top = peak_response(&mut transformer, 20_000.0, 0.1) / 0.1;
        assert!(top < mid * 0.35, "20 kHz was only attenuated to {top}");
    }

    /// Total harmonic distortion of a 60 Hz tone, as a percentage.
    ///
    /// Peak gain is a poor probe for core saturation: the cubic magnetizing
    /// term is nearly in quadrature with the fundamental, so it barely shifts
    /// the peak while adding clearly measurable odd-harmonic energy.
    fn thd_percent(transformer: &mut OutputTransformer, amplitude: f32) -> f64 {
        transformer.reset();
        let frequency = 60.0f64;
        for n in 0..8_192 {
            let x = amplitude * (std::f32::consts::TAU * 60.0 * n as f32 / FS).sin();
            transformer.process(x);
        }
        let count = 16_384usize;
        let mut samples = Vec::with_capacity(count);
        for n in 0..count {
            let x = amplitude * (std::f32::consts::TAU * 60.0 * (8_192 + n) as f32 / FS).sin();
            samples.push(transformer.process(x) as f64);
        }

        let magnitude = |freq: f64| -> f64 {
            let (mut real, mut imag) = (0.0, 0.0);
            for (index, sample) in samples.iter().enumerate() {
                let t = index as f64 / count as f64;
                let window = 0.5 - 0.5 * (std::f64::consts::TAU * t).cos();
                let phase = std::f64::consts::TAU * freq * index as f64 / FS as f64;
                real += sample * window * phase.cos();
                imag += sample * window * phase.sin();
            }
            (real * real + imag * imag).sqrt() / count as f64
        };

        let fundamental = magnitude(frequency).max(1.0e-12);
        let mut harmonics = 0.0;
        for order in 2..=9 {
            let level = magnitude(frequency * order as f64);
            harmonics += level * level;
        }
        100.0 * harmonics.sqrt() / fundamental
    }

    #[test]
    fn core_saturates_on_loud_low_frequency_content() {
        let mut transformer = prepared();
        let quiet = thd_percent(&mut transformer, 0.05);
        let loud = thd_percent(&mut transformer, 1.9);
        assert!(quiet < 0.02, "core distorted at low level: {quiet}%");
        assert!(loud > 0.3, "core did not saturate when driven: {loud}%");
        assert!(
            loud > quiet * 20.0,
            "distortion did not grow with level: {quiet}% -> {loud}%"
        );
    }

    #[test]
    fn core_distortion_is_odd_ordered() {
        // A symmetric cubic reluctance curve generates third-order product, not
        // second: an even-dominant result would mean the flux state has picked
        // up a DC offset it should not have.
        let mut transformer = prepared();
        transformer.reset();
        for n in 0..8_192 {
            let x = 1.9 * (std::f32::consts::TAU * 60.0 * n as f32 / FS).sin();
            transformer.process(x);
        }
        let count = 16_384usize;
        let mut samples = Vec::with_capacity(count);
        for n in 0..count {
            let x = 1.9 * (std::f32::consts::TAU * 60.0 * (8_192 + n) as f32 / FS).sin();
            samples.push(transformer.process(x) as f64);
        }
        let magnitude = |freq: f64| -> f64 {
            let (mut real, mut imag) = (0.0, 0.0);
            for (index, sample) in samples.iter().enumerate() {
                let t = index as f64 / count as f64;
                let window = 0.5 - 0.5 * (std::f64::consts::TAU * t).cos();
                let phase = std::f64::consts::TAU * freq * index as f64 / FS as f64;
                real += sample * window * phase.cos();
                imag += sample * window * phase.sin();
            }
            (real * real + imag * imag).sqrt() / count as f64
        };
        assert!(
            magnitude(180.0) > magnitude(120.0) * 4.0,
            "expected odd-order dominance"
        );
    }

    #[test]
    fn impulse_response_is_bounded_and_decays() {
        let mut transformer = prepared();
        let first = transformer.process(1.0);
        assert!(first.is_finite());
        let mut peak = first.abs();
        let mut last = first;
        for _ in 0..(FS as usize) {
            last = transformer.process(0.0);
            peak = peak.max(last.abs());
        }
        assert!(peak <= 2.0, "impulse peaked at {peak}");
        assert!(last.abs() < 1.0e-5, "did not settle: {last}");
    }

    #[test]
    fn extreme_input_stays_finite_and_bounded() {
        let mut transformer = prepared();
        for x in [1.0e9f32, -1.0e9, f32::MAX, f32::MIN] {
            let y = transformer.process(x);
            assert!(y.is_finite(), "{x} produced {y}");
            assert!(y.abs() <= 32.0);
        }
    }

    #[test]
    fn unprepared_transformer_still_passes_signal() {
        let mut transformer = OutputTransformer::new();
        let y = transformer.process(0.5);
        assert!(y.is_finite());
    }
}
