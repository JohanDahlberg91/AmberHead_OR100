//! Uniformly-partitioned FFT cabinet convolution with an embedded 4x12
//! impulse response.
//!
//! Specification section 6, phase 4.
//!
//! # About the impulse response
//!
//! The IR is **synthesised** by [`synthesise_4x12_ir`] from a documented filter
//! cascade rather than being a sampled measurement of a physical Celestion
//! Vintage 30 cabinet. Shipping a real measurement would mean redistributing
//! someone else's copyrighted recording, so the default cab here is a model of
//! the same target: driver low-frequency tuning, cone-breakup peak, the
//! upper-midrange notch a closed 4x12 produces off-axis, the voice-coil
//! inductance rolloff, and the early cabinet reflection that gives a 4x12 its
//! comb structure. Every corner frequency, `Q` and gain is stated as a constant
//! below so the voicing can be re-tuned against a measurement.
//!
//! Because the response is generated rather than stored, it is rebuilt in
//! [`Cabinet::prepare`] whenever the host sample rate changes and is therefore
//! correct at every rate, which a fixed 48 kHz WAV would not be.
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
/// Impulse response length in samples. 1024 taps is 21 ms at 48 kHz, well past
/// the point where a close-mic'd 4x12 has decayed into the noise floor.
pub const IR_LENGTH: usize = 1024;
/// Number of uniform partitions the IR is split into.
pub const PARTITIONS: usize = IR_LENGTH / PARTITION;

/// Frequency at which the synthesised IR is normalised to unity gain, so that
/// engaging the cabinet does not change the perceived level.
const NORMALISATION_HZ: f32 = 250.0;

/// Driver/enclosure resonance: the low-frequency corner of a closed 4x12.
const RESONANCE_HZ: f32 = 85.0;
/// `Q` of that resonance; above 0.707 so it leaves the characteristic bump
/// just above the corner.
const RESONANCE_Q: f32 = 1.15;
/// Voice-coil inductance rolloff, applied as two cascaded poles for the
/// 24 dB/octave slope a guitar speaker actually shows above 5 kHz.
const ROLLOFF_HZ: f32 = 4_800.0;
const ROLLOFF_Q_FIRST: f32 = 0.54;
const ROLLOFF_Q_SECOND: f32 = 1.31;
/// Cone-breakup peak — the forward upper-midrange character of a V30.
const BREAKUP_HZ: f32 = 2_300.0;
const BREAKUP_Q: f32 = 2.4;
const BREAKUP_GAIN_DB: f32 = 5.5;
/// Off-axis cancellation notch a closed 4x12 shows in the presence region.
const NOTCH_HZ: f32 = 3_600.0;
const NOTCH_Q: f32 = 3.5;
const NOTCH_GAIN_DB: f32 = -8.0;
/// Lower-midrange dip that keeps the cab from sounding boxy.
const BODY_HZ: f32 = 430.0;
const BODY_Q: f32 = 1.1;
const BODY_GAIN_DB: f32 = -3.5;
/// Early cabinet reflection: delay in seconds and its relative level.
/// 0.36 ms is roughly the round trip across a 4x12's internal depth.
const REFLECTION_SECONDS: f32 = 0.000_36;
const REFLECTION_GAIN: f32 = -0.4;

/// Generates the default 4x12 impulse response at `sample_rate`.
///
/// The response is the impulse response of the documented biquad cascade plus a
/// single early reflection, normalised so that `|H(250 Hz)| == 1`.
pub fn synthesise_4x12_ir(sample_rate: f32) -> Vec<f32> {
    let mut chain = [
        Biquad::highpass(RESONANCE_HZ, RESONANCE_Q, sample_rate),
        Biquad::lowpass(ROLLOFF_HZ, ROLLOFF_Q_FIRST, sample_rate),
        Biquad::lowpass(ROLLOFF_HZ, ROLLOFF_Q_SECOND, sample_rate),
        Biquad::peaking(BREAKUP_HZ, BREAKUP_Q, BREAKUP_GAIN_DB, sample_rate),
        Biquad::peaking(NOTCH_HZ, NOTCH_Q, NOTCH_GAIN_DB, sample_rate),
        Biquad::peaking(BODY_HZ, BODY_Q, BODY_GAIN_DB, sample_rate),
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

    // Normalise to unity at NORMALISATION_HZ.
    let mut real = 0.0f64;
    let mut imag = 0.0f64;
    for (index, tap) in response.iter().enumerate() {
        let phase =
            -std::f64::consts::TAU * NORMALISATION_HZ as f64 * index as f64 / sample_rate as f64;
        real += *tap as f64 * phase.cos();
        imag += *tap as f64 * phase.sin();
    }
    let magnitude = (real * real + imag * imag).sqrt();
    if magnitude > 1.0e-9 {
        let scale = (1.0 / magnitude) as f32;
        for tap in response.iter_mut() {
            *tap *= scale;
        }
    }

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

    /// The generated impulse response, retained for inspection and tests.
    impulse_response: Vec<f32>,
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
        self.impulse_response = synthesise_4x12_ir(sample_rate);

        // Transform each IR partition, zero-padded into the upper half of the
        // FFT frame as overlap-save requires.
        for partition in 0..PARTITIONS {
            let mut block = vec![0.0f32; FFT_SIZE];
            for sample in 0..PARTITION {
                let source = partition * PARTITION + sample;
                if let (Some(destination), Some(value)) = (
                    block.get_mut(sample),
                    self.impulse_response.get(source).copied(),
                ) {
                    *destination = value;
                }
            }
            let start = partition * SPECTRUM_BINS;
            let Some(target) = self.ir_spectra.get_mut(start..start + SPECTRUM_BINS) else {
                self.prepared = false;
                return false;
            };
            if forward
                .process_with_scratch(&mut block, target, &mut self.fft_scratch)
                .is_err()
            {
                self.prepared = false;
                return false;
            }
        }

        self.forward = Some(forward);
        self.inverse = Some(inverse);
        self.prepared = true;
        self.reset();
        true
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
    fn ir_is_normalised_at_the_reference_frequency() {
        let ir = synthesise_4x12_ir(FS);
        let db = ir_response_db(&ir, NORMALISATION_HZ);
        assert!(db.abs() < 0.5, "250 Hz gain was {db} dB");
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
        let presence = ir_response_db(&ir, BREAKUP_HZ);
        let air = ir_response_db(&ir, 12_000.0);

        assert!(low < body - 8.0, "40 Hz only {low} dB vs body {body} dB");
        assert!(air < body - 25.0, "12 kHz only {air} dB vs body {body} dB");
        assert!(
            presence > body,
            "no cone-breakup lift: {presence} dB vs {body} dB"
        );
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
