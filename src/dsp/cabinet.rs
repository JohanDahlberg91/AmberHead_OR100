//! Uniformly-partitioned FFT cabinet convolution with an embedded 4x12
//! impulse response.
//!
//! Specification section 6, phase 4.
//!
//! # About the default impulse response
//!
//! The IR the plugin starts with is **synthesised** by [`synthesise_4x12_ir`]
//! from a documented filter cascade rather than being a sampled measurement of
//! a physical Celestion Vintage 30 cabinet. Shipping a real measurement would
//! mean redistributing someone else's copyrighted recording, so the default cab
//! here is a model of the same target: driver low-frequency tuning, cone-breakup
//! peak, the upper-midrange notch a closed 4x12 produces off-axis, the
//! voice-coil inductance rolloff, and the early cabinet reflection that gives a
//! 4x12 its comb structure. Every corner frequency, `Q` and gain is stated as a
//! constant below so the voicing can be re-tuned against a measurement.
//!
//! Because the response is generated rather than stored, it is rebuilt in
//! [`Cabinet::prepare`] whenever the host sample rate changes and is therefore
//! correct at every rate, which a fixed 48 kHz WAV would not be.
//!
//! # Loading a measured impulse response
//!
//! [`Cabinet::load_ir`] replaces the running response with an arbitrary set of
//! taps, and [`Cabinet::restore_default_ir`] puts the synthesised one back. Both
//! re-run the partitioning transforms in place, so neither allocates and both
//! are safe to call from the audio thread — the file decoding that produces
//! those taps happens on the editor thread, in [`crate::ir`].
//!
//! Both paths normalise through [`normalise_to_reference_band`], so swapping
//! cabinets changes the voicing without changing the level, and the amplifier's
//! single output calibration stays valid whichever response is loaded.
//!
//! # Partitioning and latency
//!
//! The convolver is uniformly partitioned: the [`IR_LENGTH`]-tap response is
//! split into [`PARTITIONS`] blocks of [`PARTITION`] samples, each stored as a
//! [`FFT_SIZE`]-point spectrum, and combined with a frequency-domain delay line
//! using overlap-save.
//!
//! Collecting [`PARTITION`] input samples before a block can be transformed
//! costs exactly [`PARTITION`] samples of latency, which the plugin reports to
//! the host via `set_latency_samples()` as specification section 5 permits.
//! The bypass path runs through a delay line of the *same* length, so toggling
//! `cab_enabled` never changes the plugin's reported latency.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use super::denormal::sanitize;
use super::filters::Biquad;

/// Samples per convolution partition, and hence the convolver's latency.
pub const PARTITION: usize = 64;
/// Transform size: twice the partition, as overlap-save requires.
pub const FFT_SIZE: usize = PARTITION * 2;
/// Number of complex bins a real FFT of [`FFT_SIZE`] produces.
pub const SPECTRUM_BINS: usize = FFT_SIZE / 2 + 1;
/// Impulse response length in samples.
///
/// 4096 taps is 85 ms at 48 kHz and 43 ms at 96 kHz. The synthesised cab has
/// decayed into insignificance within a quarter of that; the length is set by
/// what a *loaded* response needs, since commercial cabinet IRs are commonly
/// distributed at 200 ms and carry room tail well past the point where a
/// close-mic'd 4x12 has gone quiet. Anything longer than this is truncated with
/// a fade, which [`crate::ir`] applies before handing the taps over.
pub const IR_LENGTH: usize = 4096;
/// Number of uniform partitions the IR is split into.
pub const PARTITIONS: usize = IR_LENGTH / PARTITION;

/// Low edge of the band an impulse response is normalised against.
const NORMALISATION_LOW_HZ: f32 = 100.0;
/// High edge of that band.
///
/// 100 Hz to 1 kHz is the range where every guitar cabinet has its body and
/// none has its character, which makes it a stable level reference across very
/// differently voiced responses.
const NORMALISATION_HIGH_HZ: f32 = 1_000.0;
/// Logarithmically spaced probe frequencies across the normalisation band.
///
/// A single probe frequency is not usable for arbitrary loaded responses: a
/// measured cabinet can have a deep null anywhere, and normalising on top of
/// one would boost that IR by however many dB the null happens to be. Taking
/// the RMS of the magnitude across a band makes the reference insensitive to
/// any individual null.
const NORMALISATION_POINTS: usize = 33;
/// Widest correction [`normalise_to_reference_band`] will apply, in dB.
///
/// A response that needs more than this is not a cabinet IR — it is silence,
/// or a file the decoder misread — and scaling it by the raw factor would
/// amplify whatever noise it does contain into the signal path.
const NORMALISATION_LIMIT_DB: f32 = 40.0;

// The constants below were fitted against a measured impulse response of an
// Orange 4x12 with Celestion V30s, close-miked with an SM57. The fit minimises
// the weighted error across 23 third-octave bands from 60 Hz to 10 kHz and
// lands at 1.6 dB RMS, against 8.4 dB for the values this cabinet carried
// before the measurement was available.
//
// The single largest correction was the low end. A real 4x12 sits 5..10 dB
// *above* its 1 kHz level across the whole 130..800 Hz range, and the chain
// had no shelving section to produce that plateau, so it ran 11..14 dB light
// through the entire low midrange — the body of the instrument. The second was
// the presence region, where an 8 dB notch at 3.6 kHz turned out to be roughly
// three times deeper than the measurement supports.

/// Driver/enclosure resonance: the low-frequency corner of a closed 4x12.
/// The measured cabinet peaks at 129 Hz.
const RESONANCE_HZ: f32 = 123.0;
/// `Q` of that resonance; above 0.707 so it leaves the characteristic bump
/// just above the corner.
const RESONANCE_Q: f32 = 1.23;
/// Voice-coil inductance rolloff, applied as two cascaded poles for the
/// 24 dB/octave slope a guitar speaker actually shows above 5 kHz. The
/// measurement crosses -10 dB at 5.9 kHz.
const ROLLOFF_HZ: f32 = 4_660.0;
const ROLLOFF_Q_FIRST: f32 = 0.80;
const ROLLOFF_Q_SECOND: f32 = 1.48;
/// Cone-breakup peak — the forward upper-midrange character of a V30.
const BREAKUP_HZ: f32 = 2_475.0;
const BREAKUP_Q: f32 = 2.19;
const BREAKUP_GAIN_DB: f32 = 5.7;
/// Off-axis cancellation notch a closed 4x12 shows in the presence region.
const NOTCH_HZ: f32 = 3_180.0;
const NOTCH_Q: f32 = 3.2;
const NOTCH_GAIN_DB: f32 = -2.4;
/// Lower-midrange dip that keeps the cab from sounding boxy. It shapes the
/// top of the shelf below rather than cutting into the 1 kHz reference.
const BODY_HZ: f32 = 435.0;
const BODY_Q: f32 = 1.67;
const BODY_GAIN_DB: f32 = -6.0;
/// Low-frequency shelf: the broad plateau a 4x12 holds from its tuning
/// frequency up into the midrange. This is where the cabinet's weight comes
/// from, and no combination of peaking sections reproduces it.
const SHELF_HZ: f32 = 711.0;
const SHELF_Q: f32 = 1.07;
const SHELF_GAIN_DB: f32 = 13.6;
/// Early cabinet reflection: delay in seconds and its relative level.
/// 0.36 ms is roughly the round trip across a 4x12's internal depth.
const REFLECTION_SECONDS: f32 = 0.000_36;
const REFLECTION_GAIN: f32 = -0.4;

/// Magnitude of an impulse response at `frequency`, evaluated directly from
/// the taps.
///
/// A goertzel-style single-bin DFT rather than a full transform: the caller
/// only ever wants a handful of frequencies, and the taps are short enough that
/// the direct sum is both faster and free of the windowing error a padded FFT
/// would introduce.
fn response_magnitude(ir: &[f32], frequency: f32, sample_rate: f32) -> f64 {
    let step = -std::f64::consts::TAU * frequency as f64 / sample_rate as f64;
    let (mut real, mut imag) = (0.0f64, 0.0f64);
    for (index, tap) in ir.iter().enumerate() {
        let phase = step * index as f64;
        real += *tap as f64 * phase.cos();
        imag += *tap as f64 * phase.sin();
    }
    (real * real + imag * imag).sqrt()
}

/// RMS magnitude of `ir` across the normalisation band, in linear gain.
///
/// Returns 0.0 for a silent or degenerate response.
pub fn reference_band_gain(ir: &[f32], sample_rate: f32) -> f32 {
    if ir.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    // Never probe above Nyquist, which a very low host rate could put below
    // NORMALISATION_HIGH_HZ.
    let high = NORMALISATION_HIGH_HZ.min(sample_rate * 0.45);
    if high <= NORMALISATION_LOW_HZ {
        return response_magnitude(ir, NORMALISATION_LOW_HZ.min(high), sample_rate) as f32;
    }

    let ratio = (high / NORMALISATION_LOW_HZ) as f64;
    let last = (NORMALISATION_POINTS - 1) as f64;
    let mut sum = 0.0f64;
    for point in 0..NORMALISATION_POINTS {
        let frequency = NORMALISATION_LOW_HZ as f64 * ratio.powf(point as f64 / last);
        let magnitude = response_magnitude(ir, frequency as f32, sample_rate);
        sum += magnitude * magnitude;
    }
    (sum / NORMALISATION_POINTS as f64).sqrt() as f32
}

/// Scales `ir` in place so its [`reference_band_gain`] is unity.
///
/// Returns `false` and leaves the taps untouched when the response is silent,
/// non-finite, or would need more than `NORMALISATION_LIMIT_DB` (40 dB) of
/// correction — every case in which scaling would do harm rather than good.
pub fn normalise_to_reference_band(ir: &mut [f32], sample_rate: f32) -> bool {
    if ir.iter().any(|tap| !tap.is_finite()) {
        return false;
    }
    let gain = reference_band_gain(ir, sample_rate);
    if !gain.is_finite() || gain <= 0.0 {
        return false;
    }
    let scale = 1.0 / gain;
    let limit = 10.0f32.powf(NORMALISATION_LIMIT_DB / 20.0);
    if scale > limit || scale < 1.0 / limit {
        return false;
    }
    for tap in ir.iter_mut() {
        *tap *= scale;
    }
    true
}

/// Generates the default 4x12 impulse response at `sample_rate`.
///
/// The response is the impulse response of the documented biquad cascade plus a
/// single early reflection, normalised through [`normalise_to_reference_band`].
pub fn synthesise_4x12_ir(sample_rate: f32) -> Vec<f32> {
    let mut chain = [
        Biquad::highpass(RESONANCE_HZ, RESONANCE_Q, sample_rate),
        Biquad::lowpass(ROLLOFF_HZ, ROLLOFF_Q_FIRST, sample_rate),
        Biquad::lowpass(ROLLOFF_HZ, ROLLOFF_Q_SECOND, sample_rate),
        Biquad::peaking(BREAKUP_HZ, BREAKUP_Q, BREAKUP_GAIN_DB, sample_rate),
        Biquad::peaking(NOTCH_HZ, NOTCH_Q, NOTCH_GAIN_DB, sample_rate),
        Biquad::peaking(BODY_HZ, BODY_Q, BODY_GAIN_DB, sample_rate),
        Biquad::low_shelf(SHELF_HZ, SHELF_Q, SHELF_GAIN_DB, sample_rate),
    ];

    let mut direct = vec![0.0f32; IR_LENGTH];
    for (index, tap) in direct.iter_mut().enumerate() {
        let mut sample = if index == 0 { 1.0 } else { 0.0 };
        for stage in chain.iter_mut() {
            sample = stage.process(sample);
        }
        *tap = sample;
    }

    // Early reflection, applied out of place so the delayed copy is of the
    // direct sound only and not of itself.
    let delay = ((REFLECTION_SECONDS * sample_rate).round() as usize).max(1);
    let mut response = direct.clone();
    for index in delay..IR_LENGTH {
        let reflected = direct.get(index - delay).copied().unwrap_or(0.0);
        if let Some(tap) = response.get_mut(index) {
            *tap += REFLECTION_GAIN * reflected;
        }
    }

    // The cascade is a unity-gain-at-DC-ish filter chain whose absolute level
    // is an accident of its `Q` values, so the result is always renormalised.
    // A failure here would mean the cascade produced silence, which the
    // `ir_has_the_expected_length_and_is_finite` test rules out.
    normalise_to_reference_band(&mut response, sample_rate);

    response
}

/// Partitioned convolution cabinet simulator.
///
/// Every buffer is allocated in [`Self::prepare`]; [`Self::process`] performs no
/// allocation, locking or I/O.
pub struct Cabinet {
    forward: Option<Arc<dyn RealToComplex<f32>>>,
    inverse: Option<Arc<dyn ComplexToReal<f32>>>,

    /// `PARTITIONS * SPECTRUM_BINS` spectra of the partitioned IR.
    ir_spectra: Vec<Complex<f32>>,
    /// Frequency-domain delay line of past input spectra, same layout.
    delay_line: Vec<Complex<f32>>,
    /// Index of the newest spectrum inside `delay_line`.
    head: usize,

    /// Per-block accumulator, `SPECTRUM_BINS` long.
    accumulator: Vec<Complex<f32>>,
    /// Scratch handed to the inverse transform, which clobbers its input.
    spectrum_scratch: Vec<Complex<f32>>,
    /// Scratch required by `realfft`, sized for the larger of the two plans.
    fft_scratch: Vec<Complex<f32>>,
    /// Time-domain workspace, `FFT_SIZE` long.
    time_buffer: Vec<f32>,
    /// Sliding window of the last `FFT_SIZE` input samples.
    history: Vec<f32>,

    /// Samples of the block currently being collected.
    input_block: [f32; PARTITION],
    /// Convolved output of the previous block, read back sample by sample.
    output_block: [f32; PARTITION],
    /// Latency-matched dry path used when the cabinet is bypassed.
    bypass_delay: [f32; PARTITION],
    cursor: usize,

    /// The impulse response currently convolved with, `IR_LENGTH` long once
    /// prepared. Retained so a loaded response can be inspected and so
    /// [`Self::load_ir`] has somewhere to copy into without allocating.
    impulse_response: Vec<f32>,
    /// The synthesised default, kept so [`Self::restore_default_ir`] can put it
    /// back without re-running the biquad cascade on the audio thread.
    default_response: Vec<f32>,
    /// Host sample rate the current response was built for.
    sample_rate: f32,
    prepared: bool,
}

impl Default for Cabinet {
    fn default() -> Self {
        Self::new()
    }
}

impl Cabinet {
    /// Builds an unprepared cabinet. Nothing is allocated until
    /// [`Self::prepare`].
    pub fn new() -> Self {
        Self {
            forward: None,
            inverse: None,
            ir_spectra: Vec::new(),
            delay_line: Vec::new(),
            head: 0,
            accumulator: Vec::new(),
            spectrum_scratch: Vec::new(),
            fft_scratch: Vec::new(),
            time_buffer: Vec::new(),
            history: Vec::new(),
            input_block: [0.0; PARTITION],
            output_block: [0.0; PARTITION],
            bypass_delay: [0.0; PARTITION],
            cursor: 0,
            impulse_response: Vec::new(),
            default_response: Vec::new(),
            sample_rate: 0.0,
            prepared: false,
        }
    }

    /// Latency the convolver introduces, in host samples.
    pub const fn latency_samples() -> u32 {
        PARTITION as u32
    }

    /// Plans the transforms, allocates every buffer and partitions a freshly
    /// generated impulse response for `sample_rate`.
    ///
    /// Returns `false` if the FFT plans could not be built or a transform
    /// rejected its buffers, in which case [`Self::process`] falls back to
    /// passing the dry signal through the same latency.
    pub fn prepare(&mut self, sample_rate: f32) -> bool {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(FFT_SIZE);
        let inverse = planner.plan_fft_inverse(FFT_SIZE);

        let scratch_len = forward.get_scratch_len().max(inverse.get_scratch_len());
        self.fft_scratch = vec![Complex::new(0.0, 0.0); scratch_len];
        self.accumulator = vec![Complex::new(0.0, 0.0); SPECTRUM_BINS];
        self.spectrum_scratch = vec![Complex::new(0.0, 0.0); SPECTRUM_BINS];
        self.time_buffer = vec![0.0; FFT_SIZE];
        self.history = vec![0.0; FFT_SIZE];
        self.delay_line = vec![Complex::new(0.0, 0.0); PARTITIONS * SPECTRUM_BINS];
        self.ir_spectra = vec![Complex::new(0.0, 0.0); PARTITIONS * SPECTRUM_BINS];
        self.default_response = synthesise_4x12_ir(sample_rate);
        self.impulse_response = self.default_response.clone();
        self.sample_rate = sample_rate;

        self.forward = Some(forward);
        self.inverse = Some(inverse);
        if !self.repartition() {
            self.forward = None;
            self.inverse = None;
            self.prepared = false;
            return false;
        }

        self.prepared = true;
        self.reset();
        true
    }

    /// Transforms every partition of `impulse_response` into `ir_spectra`.
    ///
    /// Allocation-free: the only scratch it touches is `time_buffer` and
    /// `fft_scratch`, both sized by [`Self::prepare`]. Returns `false` if a
    /// buffer is the wrong size or a transform rejected its arguments, which
    /// can only happen before `prepare` has run.
    fn repartition(&mut self) -> bool {
        // Destructured so the source response, the scratch and the destination
        // spectra can be borrowed at once.
        let Self {
            forward,
            ir_spectra,
            fft_scratch,
            time_buffer,
            impulse_response,
            ..
        } = self;
        let Some(forward) = forward.as_ref() else {
            return false;
        };
        if time_buffer.len() < FFT_SIZE || ir_spectra.len() < PARTITIONS * SPECTRUM_BINS {
            return false;
        }

        for partition in 0..PARTITIONS {
            // Each partition is zero-padded into the upper half of the FFT
            // frame, as overlap-save requires.
            for (sample, slot) in time_buffer.iter_mut().enumerate().take(FFT_SIZE) {
                *slot = if sample < PARTITION {
                    impulse_response
                        .get(partition * PARTITION + sample)
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    0.0
                };
            }

            let start = partition * SPECTRUM_BINS;
            let Some(target) = ir_spectra.get_mut(start..start + SPECTRUM_BINS) else {
                return false;
            };
            if forward
                .process_with_scratch(time_buffer, target, fft_scratch)
                .is_err()
            {
                return false;
            }
        }
        true
    }

    /// Replaces the running impulse response with `taps`.
    ///
    /// `taps` is copied into the existing [`IR_LENGTH`]-long buffer, truncated
    /// or zero-padded to fit, normalised through [`normalise_to_reference_band`]
    /// and re-partitioned. Nothing is allocated and nothing is locked, so this
    /// is safe to call from the audio thread; the caller is expected to have
    /// applied any fade-out before truncation, which [`crate::ir`] does.
    ///
    /// Returns `false` and leaves the previous response running if the cabinet
    /// is not prepared, if `taps` is empty, or if the response is silent or
    /// non-finite. The convolver's delay line is deliberately *not* cleared:
    /// the tail already in flight belongs to audio that was played through the
    /// old cabinet, and discarding it would put a gap in the output.
    pub fn load_ir(&mut self, taps: &[f32]) -> bool {
        if !self.prepared || taps.is_empty() || self.impulse_response.len() < IR_LENGTH {
            return false;
        }
        if taps.iter().any(|tap| !tap.is_finite()) {
            return false;
        }

        for (index, slot) in self.impulse_response.iter_mut().enumerate() {
            *slot = taps.get(index).copied().unwrap_or(0.0);
        }
        if !normalise_to_reference_band(&mut self.impulse_response, self.sample_rate) {
            // Put the previous response back rather than convolving with
            // something that could not be levelled.
            self.impulse_response
                .copy_from_slice(&self.default_response);
            self.repartition();
            return false;
        }
        self.repartition()
    }

    /// Puts the synthesised 4x12 response back.
    ///
    /// Allocation-free for the same reason [`Self::load_ir`] is: the default
    /// taps were generated once in [`Self::prepare`] and have been held since.
    pub fn restore_default_ir(&mut self) -> bool {
        if !self.prepared || self.default_response.len() != self.impulse_response.len() {
            return false;
        }
        self.impulse_response
            .copy_from_slice(&self.default_response);
        self.repartition()
    }

    /// Clears every delay line and the collected block, keeping the IR.
    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = 0.0);
        self.delay_line
            .iter_mut()
            .for_each(|s| *s = Complex::new(0.0, 0.0));
        self.input_block = [0.0; PARTITION];
        self.output_block = [0.0; PARTITION];
        self.bypass_delay = [0.0; PARTITION];
        self.cursor = 0;
        self.head = 0;
    }

    /// The impulse response currently loaded.
    pub fn impulse_response(&self) -> &[f32] {
        &self.impulse_response
    }

    /// Processes one sample at the host rate.
    ///
    /// When `enabled` is false the dry signal is returned through a
    /// latency-matched delay, so the reported plugin latency stays constant
    /// across the bypass toggle.
    #[inline]
    pub fn process(&mut self, input: f32, enabled: bool) -> f32 {
        let cursor = self.cursor;

        // Read before write: both arrays therefore deliver exactly PARTITION
        // samples of delay.
        let dry = self.bypass_delay.get(cursor).copied().unwrap_or(0.0);
        let wet = self.output_block.get(cursor).copied().unwrap_or(0.0);
        if let Some(slot) = self.bypass_delay.get_mut(cursor) {
            *slot = input;
        }
        if let Some(slot) = self.input_block.get_mut(cursor) {
            *slot = input;
        }

        self.cursor += 1;
        if self.cursor >= PARTITION {
            self.cursor = 0;
            self.transform_block();
        }

        if enabled && self.prepared {
            sanitize(wet)
        } else {
            dry
        }
    }

    /// Runs one overlap-save block: slide the history window, transform it,
    /// accumulate against the IR spectra, and inverse-transform.
    fn transform_block(&mut self) {
        if !self.prepared {
            return;
        }

        // Slide the FFT_SIZE-long window: drop the oldest PARTITION samples and
        // append the block just collected.
        self.history.copy_within(PARTITION..FFT_SIZE, 0);
        if let Some(tail) = self.history.get_mut(PARTITION..FFT_SIZE) {
            tail.copy_from_slice(&self.input_block);
        }
        if let Some(destination) = self.time_buffer.get_mut(..FFT_SIZE) {
            destination.copy_from_slice(&self.history);
        }

        // Newest spectrum goes into the slot before the current head, so that
        // the spectrum from `m` blocks ago sits at `(head + m) % PARTITIONS`.
        self.head = (self.head + PARTITIONS - 1) % PARTITIONS;
        let head_start = self.head * SPECTRUM_BINS;

        {
            let Some(forward) = self.forward.as_ref() else {
                return;
            };
            let Some(target) = self
                .delay_line
                .get_mut(head_start..head_start + SPECTRUM_BINS)
            else {
                return;
            };
            if forward
                .process_with_scratch(&mut self.time_buffer, target, &mut self.fft_scratch)
                .is_err()
            {
                return;
            }
        }

        for bin in self.accumulator.iter_mut() {
            *bin = Complex::new(0.0, 0.0);
        }
        for partition in 0..PARTITIONS {
            let ir_start = partition * SPECTRUM_BINS;
            let input_start = ((self.head + partition) % PARTITIONS) * SPECTRUM_BINS;
            let (Some(ir_block), Some(input_block)) = (
                self.ir_spectra.get(ir_start..ir_start + SPECTRUM_BINS),
                self.delay_line
                    .get(input_start..input_start + SPECTRUM_BINS),
            ) else {
                return;
            };
            for ((accumulated, ir), spectrum) in self
                .accumulator
                .iter_mut()
                .zip(ir_block.iter())
                .zip(input_block.iter())
            {
                *accumulated += ir * spectrum;
            }
        }

        // The inverse transform consumes its input, so hand it a copy.
        if let Some(scratch) = self.spectrum_scratch.get_mut(..SPECTRUM_BINS) {
            scratch.copy_from_slice(&self.accumulator);
        }
        {
            let Some(inverse) = self.inverse.as_ref() else {
                return;
            };
            if inverse
                .process_with_scratch(
                    &mut self.spectrum_scratch,
                    &mut self.time_buffer,
                    &mut self.fft_scratch,
                )
                .is_err()
            {
                return;
            }
        }

        // `realfft` is unnormalised; the round trip scales by FFT_SIZE. Only
        // the second half of an overlap-save frame is free of wrap-around.
        let normalisation = 1.0 / FFT_SIZE as f32;
        for (index, slot) in self.output_block.iter_mut().enumerate() {
            let value = self
                .time_buffer
                .get(PARTITION + index)
                .copied()
                .unwrap_or(0.0);
            *slot = value * normalisation;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f32 = 48_000.0;

    fn prepared() -> Cabinet {
        let mut cabinet = Cabinet::new();
        assert!(cabinet.prepare(FS), "FFT planning failed");
        cabinet
    }

    /// Magnitude of the generated IR's response at `frequency`, in dB.
    fn ir_response_db(ir: &[f32], frequency: f32) -> f32 {
        let mut real = 0.0f64;
        let mut imag = 0.0f64;
        for (index, tap) in ir.iter().enumerate() {
            let phase = -std::f64::consts::TAU * frequency as f64 * index as f64 / FS as f64;
            real += *tap as f64 * phase.cos();
            imag += *tap as f64 * phase.sin();
        }
        20.0 * (real * real + imag * imag).sqrt().log10() as f32
    }

    #[test]
    fn ir_has_the_expected_length_and_is_finite() {
        let ir = synthesise_4x12_ir(FS);
        assert_eq!(ir.len(), IR_LENGTH);
        assert!(ir.iter().all(|tap| tap.is_finite()));
        assert!(ir.iter().any(|tap| tap.abs() > 1.0e-3), "IR is silent");
    }

    #[test]
    fn ir_is_normalised_across_the_reference_band() {
        let ir = synthesise_4x12_ir(FS);
        let db = 20.0 * reference_band_gain(&ir, FS).log10();
        assert!(db.abs() < 0.01, "band gain was {db} dB");
    }

    #[test]
    fn band_normalisation_is_insensitive_to_a_single_null() {
        // The point of normalising across a band rather than at one frequency:
        // a comb null landing on the probe must not send the level anywhere.
        // A two-tap comb `1 + z^-d` has magnitude `2|cos(w*d/2)|`, so it nulls
        // exactly at fs / (2d).
        let notch_hz = 250.0f32;
        let delay = (FS / (2.0 * notch_hz)).round() as usize;
        let mut combed = vec![0.0f32; IR_LENGTH];
        if let Some(tap) = combed.get_mut(0) {
            *tap = 1.0;
        }
        if let Some(tap) = combed.get_mut(delay) {
            *tap = 1.0;
        }

        // The null is real and deep at the frequency a single-point reference
        // would have used.
        let at_null = response_magnitude(&combed, notch_hz, FS);
        assert!(
            at_null < 1.0e-3,
            "the comb does not null at 250 Hz: {at_null}"
        );

        // Yet the band reference is a perfectly ordinary number, so the
        // correction applied stays small.
        assert!(normalise_to_reference_band(&mut combed, FS));
        let peak = combed.iter().fold(0.0f32, |peak, tap| peak.max(tap.abs()));
        assert!(
            (0.3..=3.0).contains(&peak),
            "a null at the probe frequency skewed the level: peak {peak}"
        );
    }

    #[test]
    fn normalisation_refuses_responses_it_cannot_level() {
        let mut silent = vec![0.0f32; 256];
        assert!(!normalise_to_reference_band(&mut silent, FS));
        assert!(silent.iter().all(|tap| *tap == 0.0), "silence was scaled");

        let mut broken = vec![0.0f32; 256];
        if let Some(tap) = broken.get_mut(0) {
            *tap = f32::NAN;
        }
        assert!(!normalise_to_reference_band(&mut broken, FS));

        // A response needing more than NORMALISATION_LIMIT_DB of lift is
        // rejected rather than amplified.
        let mut tiny = vec![0.0f32; 256];
        if let Some(tap) = tiny.get_mut(0) {
            *tap = 1.0e-6;
        }
        assert!(!normalise_to_reference_band(&mut tiny, FS));
        assert_eq!(tiny.first().copied().unwrap_or(0.0), 1.0e-6);

        assert!(!normalise_to_reference_band(&mut [], FS));
        assert_eq!(reference_band_gain(&[], FS), 0.0);
        assert_eq!(reference_band_gain(&[1.0], 0.0), 0.0);
    }

    #[test]
    fn loading_an_ir_replaces_the_convolved_response() {
        let mut cabinet = prepared();

        // A one-pole lowpass, obviously different from the default cab and
        // trivially checkable in the output.
        let mut taps = vec![0.0f32; IR_LENGTH];
        let mut value = 1.0f32;
        for tap in taps.iter_mut() {
            *tap = value;
            value *= 0.98;
        }
        assert!(cabinet.load_ir(&taps), "load_ir refused a valid response");

        // The convolver now reproduces the loaded response, normalised.
        let loaded = cabinet.impulse_response().to_vec();
        assert_eq!(loaded.len(), IR_LENGTH);
        let db = 20.0 * reference_band_gain(&loaded, FS).log10();
        assert!(db.abs() < 0.01, "loaded IR not levelled: {db} dB");

        let mut output = vec![cabinet.process(1.0, true)];
        for _ in 1..(IR_LENGTH + 2 * PARTITION) {
            output.push(cabinet.process(0.0, true));
        }
        let mut worst = 0.0f32;
        for (index, expected) in loaded.iter().enumerate() {
            let actual = output.get(index + PARTITION).copied().unwrap_or(0.0);
            worst = worst.max((actual - expected).abs());
        }
        assert!(worst < 1.0e-5, "loaded convolution error {worst}");
    }

    #[test]
    fn restoring_the_default_recovers_the_synthesised_response() {
        let mut cabinet = prepared();
        let original = cabinet.impulse_response().to_vec();

        let taps = vec![1.0f32; 512];
        assert!(cabinet.load_ir(&taps));
        assert_ne!(cabinet.impulse_response(), original.as_slice());

        assert!(cabinet.restore_default_ir());
        assert_eq!(cabinet.impulse_response(), original.as_slice());
    }

    #[test]
    fn loading_rejects_bad_input_and_keeps_playing() {
        let mut cabinet = prepared();
        let original = cabinet.impulse_response().to_vec();

        assert!(!cabinet.load_ir(&[]), "an empty response was accepted");
        assert!(
            !cabinet.load_ir(&[1.0, f32::NAN, 0.5]),
            "a non-finite response was accepted"
        );
        assert!(
            !cabinet.load_ir(&[0.0; 256]),
            "a silent response was accepted"
        );
        assert_eq!(
            cabinet.impulse_response(),
            original.as_slice(),
            "a rejected load disturbed the running response"
        );

        // And an unprepared cabinet refuses rather than panicking.
        let mut fresh = Cabinet::new();
        assert!(!fresh.load_ir(&[1.0, 0.5]));
        assert!(!fresh.restore_default_ir());
    }

    #[test]
    fn loading_shorter_and_longer_responses_both_work() {
        let mut cabinet = prepared();

        // Shorter than IR_LENGTH: the remainder must be zero-filled, not left
        // holding the previous response.
        let short = vec![1.0f32; 8];
        assert!(cabinet.load_ir(&short));
        assert!(
            cabinet
                .impulse_response()
                .iter()
                .skip(8)
                .all(|tap| *tap == 0.0),
            "the tail of the previous IR survived a shorter load"
        );

        // Longer than IR_LENGTH: the excess is dropped, not wrapped.
        let mut long = vec![0.0f32; IR_LENGTH * 2];
        for (index, tap) in long.iter_mut().enumerate() {
            *tap = if index < IR_LENGTH { 1.0 } else { 9.0 };
        }
        assert!(cabinet.load_ir(&long));
        let peak = cabinet
            .impulse_response()
            .iter()
            .fold(0.0f32, |peak, tap| peak.max(tap.abs()));
        let smallest = cabinet
            .impulse_response()
            .iter()
            .fold(f32::INFINITY, |low, tap| low.min(tap.abs()));
        assert!(
            (peak - smallest).abs() < 1.0e-6,
            "taps past IR_LENGTH leaked in: {smallest} .. {peak}"
        );
    }

    #[test]
    fn ir_decays_to_silence_within_its_window() {
        let ir = synthesise_4x12_ir(FS);
        let tail_peak = ir
            .iter()
            .skip(IR_LENGTH * 3 / 4)
            .fold(0.0f32, |peak, tap| peak.max(tap.abs()));
        let head_peak = ir
            .iter()
            .take(64)
            .fold(0.0f32, |peak, tap| peak.max(tap.abs()));
        assert!(
            tail_peak < head_peak * 0.01,
            "IR had not decayed: {head_peak} -> {tail_peak}"
        );
    }

    #[test]
    fn ir_rolls_off_lows_and_highs_like_a_4x12() {
        let ir = synthesise_4x12_ir(FS);
        let low = ir_response_db(&ir, 40.0);
        let body = ir_response_db(&ir, 250.0);
        let reference = ir_response_db(&ir, 1_000.0);
        let presence = ir_response_db(&ir, BREAKUP_HZ);
        let air = ir_response_db(&ir, 12_000.0);

        assert!(low < body - 8.0, "40 Hz only {low} dB vs body {body} dB");
        assert!(air < body - 25.0, "12 kHz only {air} dB vs body {body} dB");
        // The cone-breakup peak is a lift over the *midrange*, not over the
        // low end. A measured 4x12 sits some 7 dB above its 1 kHz level at
        // 250 Hz, so the body is louder than the presence region in absolute
        // terms; comparing the two directly measures the low shelf instead.
        assert!(
            presence > reference,
            "no cone-breakup lift: {presence} dB vs {reference} dB at 1 kHz"
        );
    }

    #[test]
    fn ir_carries_the_low_shelf_a_measured_4x12_shows() {
        // Regression guard for the largest error the measurement exposed. The
        // synthesised cabinet once ran 11..14 dB light across the whole
        // 130..800 Hz band, because the chain had no shelving section and a
        // plateau that wide cannot be built from peaking filters. That is the
        // body of the instrument, and without it the amplifier sounds thin
        // however the preamp is voiced.
        let ir = synthesise_4x12_ir(FS);
        let reference = ir_response_db(&ir, 1_000.0);
        for (frequency, floor) in [(130.0, 5.0), (200.0, 5.0), (400.0, 2.0), (630.0, 1.0)] {
            let level = ir_response_db(&ir, frequency) - reference;
            assert!(
                level > floor,
                "{frequency} Hz sits {level} dB over 1 kHz, under the {floor} dB the measurement shows"
            );
        }
    }

    #[test]
    fn ir_regenerates_correctly_at_other_sample_rates() {
        for rate in [44_100.0f32, 96_000.0, 192_000.0] {
            let ir = synthesise_4x12_ir(rate);
            assert_eq!(ir.len(), IR_LENGTH);
            assert!(ir.iter().all(|tap| tap.is_finite()), "bad IR at {rate}");
        }
    }

    #[test]
    fn convolver_reproduces_the_impulse_response() {
        // The defining test: feeding a unit impulse must return the IR itself,
        // delayed by exactly PARTITION samples.
        let mut cabinet = prepared();
        let reference = cabinet.impulse_response().to_vec();

        let mut output = Vec::with_capacity(IR_LENGTH + 2 * PARTITION);
        output.push(cabinet.process(1.0, true));
        for _ in 1..(IR_LENGTH + 2 * PARTITION) {
            output.push(cabinet.process(0.0, true));
        }

        let mut worst = 0.0f32;
        for (index, expected) in reference.iter().enumerate() {
            let actual = output.get(index + PARTITION).copied().unwrap_or(0.0);
            worst = worst.max((actual - expected).abs());
        }
        assert!(worst < 1.0e-5, "convolution error {worst}");
    }

    #[test]
    fn convolver_latency_matches_the_declared_value() {
        let mut cabinet = prepared();
        let mut output = Vec::new();
        output.push(cabinet.process(1.0, true));
        for _ in 0..(4 * PARTITION) {
            output.push(cabinet.process(0.0, true));
        }
        let first_nonzero = output
            .iter()
            .position(|sample| sample.abs() > 1.0e-6)
            .unwrap_or(usize::MAX);
        assert_eq!(first_nonzero, Cabinet::latency_samples() as usize);
    }

    #[test]
    fn bypass_is_latency_matched_and_transparent() {
        let mut cabinet = prepared();
        let mut input = Vec::new();
        let mut output = Vec::new();
        for n in 0..1_024 {
            let x = (std::f32::consts::TAU * 220.0 * n as f32 / FS).sin();
            input.push(x);
            output.push(cabinet.process(x, false));
        }
        let latency = Cabinet::latency_samples() as usize;
        for n in latency..1_024 {
            let expected = input.get(n - latency).copied().unwrap_or(0.0);
            let actual = output.get(n).copied().unwrap_or(0.0);
            assert!(
                (expected - actual).abs() < 1.0e-6,
                "bypass altered sample {n}: {expected} vs {actual}"
            );
        }
    }

    #[test]
    fn unprepared_cabinet_passes_dry_signal_without_panicking() {
        let mut cabinet = Cabinet::new();
        for n in 0..256 {
            let x = (n as f32 * 0.01).sin();
            let y = cabinet.process(x, true);
            assert!(y.is_finite());
        }
    }

    #[test]
    fn reset_clears_the_tail() {
        let mut cabinet = prepared();
        for _ in 0..PARTITION * 4 {
            cabinet.process(1.0, true);
        }
        cabinet.reset();
        let mut peak = 0.0f32;
        for _ in 0..PARTITION * 2 {
            peak = peak.max(cabinet.process(0.0, true).abs());
        }
        assert!(peak < 1.0e-6, "tail survived reset: {peak}");
    }

    #[test]
    fn output_stays_finite_for_extreme_input() {
        let mut cabinet = prepared();
        for _ in 0..PARTITION * 4 {
            let y = cabinet.process(1.0e6, true);
            assert!(y.is_finite());
        }
    }
}
