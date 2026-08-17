//! Complete OR100 signal chain.
//!
//! Implements the block diagram in specification section 2:
//!
//! ```text
//! In -> [8x Up] -> [Channel Switch]
//!         Clean: V1 -> 2-band EQ -> Clean Volume
//!         Dirty: V2 -> V3 -> V4 (Gain Boost) -> 3-band EQ -> Dirty Volume
//!       -> [Global Boost +3 dB] -> [LTP PI] -> [EL34 Push-Pull] <- [B+ Sag]
//!       -> [8x Down] -> [Output Transformer] -> [Partitioned FFT Cab IR] -> Out
//! ```
//!
//! Everything from the channel switch to the power stage runs inside the 8x
//! oversampled sub-pipeline; the transformer and cabinet run at the host rate,
//! as the specification's diagram places them after the downsampler.
//!
//! # Gain staging
//!
//! Voltages here are real circuit volts, not normalized audio. The chain is
//! anchored at both ends: [`INPUT_DRIVE_VOLTS`] maps a full-scale sample onto a
//! hot humbucker's peak grid voltage, and [`OUTPUT_CALIBRATION`] scales the
//! transformer secondary back into the host's normalized range. The interstage
//! constants between them are the fixed dividers a real amplifier uses to keep
//! each successive grid inside its operating window.

use super::cabinet::Cabinet;
use super::denormal::sanitize;
use super::filters::SwitchRamp;
use super::oversampling::Oversampler8x;
use super::power::{PhaseInverter, PowerAmp, PowerMode};
use super::tonestack::{ToneStack, ToneStackCircuit};
use super::transformer::OutputTransformer;
use super::triode::{StageCircuit, Triode};
use super::{audio_taper, knob_to_rotation};

/// Grid volts a full-scale (1.0) input sample corresponds to.
///
/// A hot humbucker into a 1 MΩ input presents roughly 350 mV of peak signal, so
/// a track normalized to 0 dBFS drives the first stage exactly as a guitar
/// plugged straight in would.
pub const INPUT_DRIVE_VOLTS: f32 = 0.35;

/// Final scaling from the transformer secondary into the host's normalized
/// range, chosen so that the default patch (dirty channel, gain 6.5,
/// volume 5.0, 100 W) peaks near -10 dBFS on a full-scale input.
pub const OUTPUT_CALIBRATION: f32 = 0.34;

/// Divider between V1's plate and the clean channel's tone stack.
const CLEAN_STACK_DRIVE: f32 = 0.35;
/// Makeup applied after the clean tone stack's insertion loss, so the clean
/// channel can still reach power-stage breakup at high volume settings.
///
/// Calibrated so the clean channel stays clean at its middle volume setting —
/// a full-scale input at volume 5 lands near -18 dBFS with the power stage
/// barely working — and only reaches power-stage breakup towards volume 10.
const CLEAN_MAKEUP: f32 = 6.0;

/// Fixed divider between V2's plate and the gain pot.
const V2_TO_GAIN_POT: f32 = 0.5;
/// Divider between V3's plate and the following stage.
const V3_TO_V4: f32 = 0.06;
/// Attenuation into V4 when the gain boost stage is engaged.
const V4_DRIVE: f32 = 0.09;
/// Trim on the path that skips V4, set so engaging the boost is a musical
/// step of roughly +8 dB rather than the +30 dB the raw stage would give.
const BOOST_BYPASS_TRIM: f32 = 1.7;
/// Divider between the last preamp plate and the dirty tone stack.
const DIRTY_STACK_DRIVE: f32 = 0.30;
/// Makeup after the dirty tone stack's insertion loss.
///
/// Calibrated so the default patch drives the power stage firmly into its
/// non-linear region without pinning it: at 26.0 the EL34 stage limits so hard
/// that the tone controls lose all audible authority, which a real amplifier
/// does not do.
const DIRTY_MAKEUP: f32 = 14.0;

/// Linear gain of the +3 dB global boost.
const GLOBAL_BOOST_GAIN: f32 = 1.412_537_5;

/// Crossfade time for the channel switch and the two boost switches.
/// Specification section 3 asks for 10 ms on `global_boost`; the same ramp
/// keeps the channel and gain-boost switches click-free.
const SWITCH_RAMP_MS: f32 = 10.0;

/// Seconds for the `B+` rail to come up after leaving standby.
const WARMUP_SECONDS: f32 = 2.0;
/// Seconds for the rail to bleed away when switching to standby or off.
const RAIL_DECAY_SECONDS: f32 = 0.6;

/// How often the tone stack matrices are re-solved, in samples at the host
/// rate. 32 samples is 0.67 ms at 48 kHz, far finer than the 20 ms smoothing on
/// the EQ parameters themselves.
const CONTROL_INTERVAL: u32 = 32;

/// Which preamp channel is selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Single-stage clean channel with the 2-band tone stack.
    Clean,
    /// Three-stage cascaded channel with the 3-band tone stack.
    Dirty,
}

/// Position of the power / standby switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerState {
    /// Mains off: no rail, no heaters, lamp dark.
    Off,
    /// Heaters on, `B+` disconnected: lamp lit, no output.
    Standby,
    /// Fully operational.
    On,
}

/// Per-sample control values handed to [`AmpEngine::process_sample`].
///
/// The continuous fields carry the host's smoothed values so automation is
/// sample-accurate; the discrete fields only take effect at the next
/// control-rate boundary or through a switch ramp.
#[derive(Debug, Clone, Copy)]
pub struct SampleControls {
    /// Selected channel.
    pub channel: Channel,
    /// Clean volume, 0.0..=10.0.
    pub clean_volume: f32,
    /// Clean bass, 0.0..=10.0.
    pub clean_bass: f32,
    /// Clean treble, 0.0..=10.0.
    pub clean_treble: f32,
    /// Dirty gain, 0.0..=10.0.
    pub dirty_gain: f32,
    /// Dirty bass, 0.0..=10.0.
    pub dirty_bass: f32,
    /// Dirty middle, 0.0..=10.0.
    pub dirty_middle: f32,
    /// Dirty treble, 0.0..=10.0.
    pub dirty_treble: f32,
    /// Dirty volume, 0.0..=10.0.
    pub dirty_volume: f32,
    /// Extra cascaded gain stage.
    pub gain_boost: bool,
    /// +3 dB into the phase inverter.
    pub global_boost: bool,
    /// Power / standby switch position.
    pub power: PowerState,
    /// Wattage mode.
    pub output_power: PowerMode,
    /// Cabinet simulation on or bypassed.
    pub cab_enabled: bool,
}

impl Default for SampleControls {
    fn default() -> Self {
        Self {
            channel: Channel::Dirty,
            clean_volume: 5.0,
            clean_bass: 5.0,
            clean_treble: 5.0,
            dirty_gain: 6.5,
            dirty_bass: 5.0,
            dirty_middle: 5.0,
            dirty_treble: 5.0,
            dirty_volume: 5.0,
            gain_boost: false,
            global_boost: false,
            power: PowerState::On,
            output_power: PowerMode::Watt100,
            cab_enabled: true,
        }
    }
}

/// The complete amplifier.
#[repr(align(64))]
pub struct AmpEngine {
    oversampler: Oversampler8x,

    clean_v1: Triode,
    clean_stack: ToneStack,

    dirty_v2: Triode,
    dirty_v3: Triode,
    dirty_v4: Triode,
    dirty_stack: ToneStack,

    phase_inverter: PhaseInverter,
    power_amp: PowerAmp,
    transformer: OutputTransformer,
    cabinet: Cabinet,

    channel_ramp: SwitchRamp,
    gain_boost_ramp: SwitchRamp,
    global_boost_ramp: SwitchRamp,

    /// Fraction of the nominal `B+` currently available.
    rail: f32,
    /// Per-sample increment while warming up.
    rail_rise: f32,
    /// Per-sample decrement while bleeding down.
    rail_fall: f32,

    control_counter: u32,
    sample_rate: f32,
    prepared: bool,
}

impl Default for AmpEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AmpEngine {
    /// Builds an unprepared engine. No tables or buffers exist until
    /// [`Self::prepare`] runs.
    pub fn new() -> Self {
        Self {
            oversampler: Oversampler8x::default(),
            clean_v1: Triode::new(StageCircuit::classic_gain_stage()),
            clean_stack: ToneStack::new(ToneStackCircuit::or100_clean()),
            dirty_v2: Triode::new(StageCircuit::classic_gain_stage()),
            dirty_v3: Triode::new(StageCircuit::cascade_stage()),
            dirty_v4: Triode::new(StageCircuit::cascade_stage()),
            dirty_stack: ToneStack::new(ToneStackCircuit::or100_dirty()),
            phase_inverter: PhaseInverter::new(),
            power_amp: PowerAmp::new(),
            transformer: OutputTransformer::new(),
            cabinet: Cabinet::new(),
            channel_ramp: SwitchRamp::default(),
            gain_boost_ramp: SwitchRamp::default(),
            global_boost_ramp: SwitchRamp::default(),
            rail: 0.0,
            rail_rise: 1.0,
            rail_fall: 1.0,
            control_counter: 0,
            sample_rate: 48_000.0,
            prepared: false,
        }
    }

    /// Allocates and computes everything the audio path needs.
    ///
    /// Builds four triode load-line tables plus the two inside the phase
    /// inverter, designs the oversampling filters, solves both tone stacks,
    /// and plans the convolution FFTs. Call from `Plugin::initialize`.
    ///
    /// Returns `false` if the cabinet's FFT plans could not be built; the
    /// engine remains usable and simply passes the cabinet stage through.
    pub fn prepare(&mut self, sample_rate: f32, controls: &SampleControls) -> bool {
        self.sample_rate = sample_rate;
        let oversampled_rate = sample_rate * super::OVERSAMPLING_FACTOR as f32;

        self.oversampler.prepare();
        self.clean_v1.prepare(oversampled_rate);
        self.dirty_v2.prepare(oversampled_rate);
        self.dirty_v3.prepare(oversampled_rate);
        self.dirty_v4.prepare(oversampled_rate);
        self.phase_inverter.prepare(oversampled_rate);
        self.power_amp.prepare(oversampled_rate);
        self.transformer.prepare(sample_rate);

        self.clean_stack.prepare(
            oversampled_rate,
            knob_to_rotation(controls.clean_treble),
            knob_to_rotation(controls.clean_bass),
            ToneStack::CLEAN_FIXED_MID,
        );
        self.dirty_stack.prepare(
            oversampled_rate,
            knob_to_rotation(controls.dirty_treble),
            knob_to_rotation(controls.dirty_bass),
            knob_to_rotation(controls.dirty_middle),
        );

        self.channel_ramp.prepare(
            SWITCH_RAMP_MS,
            sample_rate,
            controls.channel == Channel::Dirty,
        );
        self.gain_boost_ramp
            .prepare(SWITCH_RAMP_MS, sample_rate, controls.gain_boost);
        self.global_boost_ramp
            .prepare(SWITCH_RAMP_MS, sample_rate, controls.global_boost);

        self.rail_rise = 1.0 / (WARMUP_SECONDS * sample_rate).max(1.0);
        self.rail_fall = 1.0 / (RAIL_DECAY_SECONDS * sample_rate).max(1.0);
        self.rail = if controls.power == PowerState::On {
            1.0
        } else {
            0.0
        };

        self.power_amp.set_mode(controls.output_power);
        // Must precede `reset()`, which seeds the reported B+ from the rail.
        self.power_amp.set_rail_availability(self.rail);
        self.control_counter = 0;
        self.prepared = true;

        let cabinet_ready = self.cabinet.prepare(sample_rate);
        self.reset();
        cabinet_ready
    }

    /// Returns every stage to its quiescent state without rebuilding tables.
    pub fn reset(&mut self) {
        self.oversampler.reset();
        self.clean_v1.reset();
        self.clean_stack.reset();
        self.dirty_v2.reset();
        self.dirty_v3.reset();
        self.dirty_v4.reset();
        self.dirty_stack.reset();
        self.phase_inverter.reset();
        self.power_amp.reset();
        self.transformer.reset();
        self.cabinet.reset();
        self.control_counter = 0;
    }

    /// Total algorithmic latency in host samples: the oversampling cascade plus
    /// the convolution partition.
    pub fn latency_samples(&self) -> u32 {
        self.oversampler.latency_samples() + Cabinet::latency_samples()
    }

    /// Instantaneous `B+` rail voltage, for the jewel lamp.
    pub fn b_plus(&self) -> f32 {
        self.power_amp.b_plus()
    }

    /// Jewel lamp brightness in `0.0..=1.0`.
    ///
    /// Dark when the amp is off, at a fixed low glow on standby (the pilot lamp
    /// runs off the heater supply), and modulated by the sagging rail when the
    /// amp is running — which is what makes it pulse with hard playing, as
    /// specification section 4 asks for.
    pub fn lamp_brightness(&self, power: PowerState) -> f32 {
        match power {
            PowerState::Off => 0.0,
            PowerState::Standby => 0.55,
            PowerState::On => {
                let nominal = self.power_amp.mode().nominal_b_plus().max(1.0);
                let ratio = (self.power_amp.b_plus() / nominal).clamp(0.0, 1.0);
                (0.60 + 0.40 * ratio).clamp(0.0, 1.0)
            }
        }
    }

    /// The impulse response the cabinet stage is currently using.
    pub fn impulse_response(&self) -> &[f32] {
        self.cabinet.impulse_response()
    }

    /// Re-solves the tone stacks and applies discrete switch positions.
    fn update_control_rate(&mut self, controls: &SampleControls) {
        self.clean_stack.set_controls(
            knob_to_rotation(controls.clean_treble),
            knob_to_rotation(controls.clean_bass),
            ToneStack::CLEAN_FIXED_MID,
        );
        self.dirty_stack.set_controls(
            knob_to_rotation(controls.dirty_treble),
            knob_to_rotation(controls.dirty_bass),
            knob_to_rotation(controls.dirty_middle),
        );
        self.power_amp.set_mode(controls.output_power);
        self.channel_ramp.set(controls.channel == Channel::Dirty);
        self.gain_boost_ramp.set(controls.gain_boost);
        self.global_boost_ramp.set(controls.global_boost);
    }

    /// Advances the `B+` rail towards the level the power switch calls for.
    #[inline(always)]
    fn advance_rail(&mut self, power: PowerState) -> f32 {
        let target = if power == PowerState::On { 1.0 } else { 0.0 };
        if self.rail < target {
            self.rail = (self.rail + self.rail_rise).min(target);
        } else if self.rail > target {
            self.rail = (self.rail - self.rail_fall).max(target);
        }
        self.rail
    }

    /// Processes one host-rate sample through the entire amplifier.
    #[inline]
    pub fn process_sample(&mut self, input: f32, controls: &SampleControls) -> f32 {
        if !self.prepared {
            return 0.0;
        }

        if self.control_counter == 0 {
            self.update_control_rate(controls);
        }
        self.control_counter += 1;
        if self.control_counter >= CONTROL_INTERVAL {
            self.control_counter = 0;
        }

        let channel_mix = self.channel_ramp.advance();
        let boost_mix = self.gain_boost_ramp.advance();
        let global_mix = self.global_boost_ramp.advance();
        let rail = self.advance_rail(controls.power);
        self.power_amp.set_rail_availability(rail);

        // Front-panel pots. Volume and gain use an audio taper, the tone
        // controls the linear taper their pots actually have.
        let clean_volume = audio_taper(knob_to_rotation(controls.clean_volume));
        let dirty_volume = audio_taper(knob_to_rotation(controls.dirty_volume));
        let gain = audio_taper(knob_to_rotation(controls.dirty_gain));
        let global_gain = 1.0 + (GLOBAL_BOOST_GAIN - 1.0) * global_mix;

        let driven = sanitize(input) * INPUT_DRIVE_VOLTS;

        // Split the borrow so the oversampler can own the closure while the
        // stages it drives stay independently mutable.
        let Self {
            oversampler,
            clean_v1,
            clean_stack,
            dirty_v2,
            dirty_v3,
            dirty_v4,
            dirty_stack,
            phase_inverter,
            power_amp,
            ..
        } = self;

        let power_output = oversampler.process(driven, |sample| {
            // --- Clean channel: V1 -> 2-band EQ -> Clean Volume -------------
            let clean = if channel_mix < 1.0 {
                let v1 = clean_v1.process(sample);
                let shaped = clean_stack.process(v1 * CLEAN_STACK_DRIVE);
                shaped * clean_volume * CLEAN_MAKEUP
            } else {
                // Fully on the dirty channel: keep the clean stages settled at
                // their quiescent point rather than leaving stale state.
                clean_v1.process(0.0);
                clean_stack.process(0.0);
                0.0
            };

            // --- Dirty channel: V2 -> V3 -> V4 -> 3-band EQ -> Volume -------
            let dirty = if channel_mix > 0.0 {
                let v2 = dirty_v2.process(sample);
                let v3 = dirty_v3.process(v2 * gain * V2_TO_GAIN_POT);
                let into_v4 = v3 * V3_TO_V4;

                // Gain boost crossfades the extra cascaded stage in and out.
                let boosted = dirty_v4.process(into_v4 * V4_DRIVE);
                let unboosted = into_v4 * BOOST_BYPASS_TRIM;
                let combined = unboosted + (boosted - unboosted) * boost_mix;

                let shaped = dirty_stack.process(combined * DIRTY_STACK_DRIVE);
                shaped * dirty_volume * DIRTY_MAKEUP
            } else {
                dirty_v2.process(0.0);
                dirty_v3.process(0.0);
                dirty_v4.process(0.0);
                dirty_stack.process(0.0);
                0.0
            };

            let preamp = clean + (dirty - clean) * channel_mix;

            // --- Global boost, phase inverter, power stage ------------------
            let driven = preamp * global_gain;
            let differential = phase_inverter.process(driven);
            power_amp.process(differential)
        });

        // --- Host rate: transformer core, then the cabinet ------------------
        let secondary = self.transformer.process(power_output);
        let cabinet = self.cabinet.process(secondary, controls.cab_enabled);

        // The power switch mutes the output entirely; the rail ramp already
        // fades the power stage, so this only guarantees hard silence when off.
        let mute = if controls.power == PowerState::On {
            1.0
        } else {
            rail
        };
        sanitize(cabinet * OUTPUT_CALIBRATION * mute)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f32 = 48_000.0;

    fn prepared_engine(controls: &SampleControls) -> AmpEngine {
        let mut engine = AmpEngine::new();
        assert!(engine.prepare(FS, controls), "cabinet preparation failed");
        engine
    }

    /// Peak output for a sine of the given amplitude, after letting the chain
    /// settle past its latency and the DC blockers.
    fn peak_for(engine: &mut AmpEngine, controls: &SampleControls, amplitude: f32, hz: f32) -> f32 {
        let settle = (FS * 0.3) as usize;
        for n in 0..settle {
            let x = amplitude * (std::f32::consts::TAU * hz * n as f32 / FS).sin();
            engine.process_sample(x, controls);
        }
        let mut peak = 0.0f32;
        for n in 0..(FS * 0.2) as usize {
            let x = amplitude * (std::f32::consts::TAU * hz * (settle + n) as f32 / FS).sin();
            peak = peak.max(engine.process_sample(x, controls).abs());
        }
        peak
    }

    #[test]
    fn latency_is_the_sum_of_its_parts() {
        let controls = SampleControls::default();
        let engine = prepared_engine(&controls);
        // 56 samples of oversampling cascade + 64 samples of convolution.
        assert_eq!(engine.latency_samples(), 56 + 64);
    }

    #[test]
    fn default_patch_lands_in_a_usable_output_window() {
        // Phase 4 gain-staging calibration: a full-scale input on the default
        // patch must produce a hot but headroom-preserving output.
        let controls = SampleControls::default();
        let mut engine = prepared_engine(&controls);
        let peak = peak_for(&mut engine, &controls, 1.0, 220.0);
        let db = 20.0 * peak.log10();
        assert!(
            (-20.0..=-1.0).contains(&db),
            "default patch peaked at {db} dBFS"
        );
    }

    /// Total harmonic distortion of a 220 Hz tone through the whole amp, as a
    /// percentage.
    ///
    /// Crest factor is unusable as a distortion proxy here: the cabinet's
    /// resonance and presence notch reshape the waveform enough to move
    /// peak-to-RMS by more than the clipping does.
    fn engine_thd_percent(controls: &SampleControls, amplitude: f32) -> f64 {
        let mut engine = prepared_engine(controls);
        let settle = (FS * 0.3) as usize;
        for n in 0..settle {
            let x = amplitude * (std::f32::consts::TAU * 220.0 * n as f32 / FS).sin();
            engine.process_sample(x, controls);
        }
        let count = 16_384usize;
        let mut samples = Vec::with_capacity(count);
        for n in 0..count {
            let x = amplitude * (std::f32::consts::TAU * 220.0 * (settle + n) as f32 / FS).sin();
            samples.push(engine.process_sample(x, controls) as f64);
        }

        let magnitude = |frequency: f64| -> f64 {
            let (mut real, mut imag) = (0.0, 0.0);
            for (index, sample) in samples.iter().enumerate() {
                let t = index as f64 / count as f64;
                let window = 0.5 - 0.5 * (std::f64::consts::TAU * t).cos();
                let phase = std::f64::consts::TAU * frequency * index as f64 / FS as f64;
                real += sample * window * phase.cos();
                imag += sample * window * phase.sin();
            }
            (real * real + imag * imag).sqrt() / count as f64
        };

        let fundamental = magnitude(220.0).max(1.0e-12);
        let mut harmonics = 0.0;
        for order in 2..=12 {
            let level = magnitude(220.0 * order as f64);
            harmonics += level * level;
        }
        100.0 * harmonics.sqrt() / fundamental
    }

    #[test]
    fn clean_channel_is_cleaner_than_the_dirty_channel() {
        let dirty_controls = SampleControls::default();
        let clean_controls = SampleControls {
            channel: Channel::Clean,
            ..SampleControls::default()
        };

        let clean = engine_thd_percent(&clean_controls, 0.2);
        let dirty = engine_thd_percent(&dirty_controls, 0.2);
        assert!(
            clean * 3.0 < dirty,
            "clean THD {clean}% is not well under dirty {dirty}%"
        );
        assert!(clean < 8.0, "clean channel distorted {clean}% at 0.2 in");
        assert!(dirty > 10.0, "dirty channel only reached {dirty}%");
    }

    #[test]
    fn distortion_tracks_the_gain_control() {
        let quiet = SampleControls {
            dirty_gain: 1.0,
            ..SampleControls::default()
        };
        let loud = SampleControls {
            dirty_gain: 9.0,
            ..SampleControls::default()
        };
        let low = engine_thd_percent(&quiet, 0.2);
        let high = engine_thd_percent(&loud, 0.2);
        assert!(low < 2.0, "gain 1 already distorting at {low}%");
        assert!(high > low * 5.0, "gain 9 only reached {high}% vs {low}%");
    }

    #[test]
    fn clean_channel_still_breaks_up_when_pushed() {
        // A real clean channel is not a limiter: slam it and it will distort.
        let controls = SampleControls {
            channel: Channel::Clean,
            clean_volume: 10.0,
            ..SampleControls::default()
        };
        let thd = engine_thd_percent(&controls, 1.0);
        assert!(thd > 10.0, "clean channel refused to break up: {thd}%");
    }

    #[test]
    fn gain_control_monotonically_increases_drive() {
        let mut previous = 0.0f32;
        for gain in [1.0f32, 3.0, 5.0, 8.0, 10.0] {
            let controls = SampleControls {
                dirty_gain: gain,
                ..SampleControls::default()
            };
            let mut engine = prepared_engine(&controls);
            let peak = peak_for(&mut engine, &controls, 0.3, 220.0);
            assert!(
                peak >= previous * 0.98,
                "gain {gain} produced {peak}, below the previous {previous}"
            );
            previous = peak;
        }
    }

    #[test]
    fn gain_boost_adds_level() {
        let plain = SampleControls::default();
        let boosted = SampleControls {
            gain_boost: true,
            ..SampleControls::default()
        };
        let mut a = prepared_engine(&plain);
        let mut b = prepared_engine(&boosted);
        let quiet = peak_for(&mut a, &plain, 0.2, 220.0);
        let loud = peak_for(&mut b, &boosted, 0.2, 220.0);
        assert!(loud > quiet, "gain boost did nothing: {quiet} -> {loud}");
    }

    #[test]
    fn global_boost_adds_roughly_three_decibels_of_drive() {
        // Measured at a low input level where nothing downstream is clipping,
        // so the +3 dB is not swallowed by compression.
        let plain = SampleControls {
            dirty_gain: 1.0,
            dirty_volume: 2.0,
            ..SampleControls::default()
        };
        let boosted = SampleControls {
            global_boost: true,
            ..plain
        };
        let mut a = prepared_engine(&plain);
        let mut b = prepared_engine(&boosted);
        let quiet = peak_for(&mut a, &plain, 0.05, 220.0);
        let loud = peak_for(&mut b, &boosted, 0.05, 220.0);
        let delta = 20.0 * (loud / quiet.max(1.0e-9)).log10();
        assert!((1.0..=4.0).contains(&delta), "global boost gave {delta} dB");
    }

    #[test]
    fn wattage_modes_are_ordered_by_level() {
        let mut levels = Vec::new();
        for mode in [
            PowerMode::Watt100,
            PowerMode::Watt70,
            PowerMode::Watt50,
            PowerMode::Watt30,
        ] {
            let controls = SampleControls {
                output_power: mode,
                dirty_gain: 3.0,
                ..SampleControls::default()
            };
            let mut engine = prepared_engine(&controls);
            levels.push(peak_for(&mut engine, &controls, 0.3, 220.0));
        }
        for pair in levels.windows(2) {
            assert!(pair[0] > pair[1], "wattage levels not ordered: {levels:?}");
        }
    }

    #[test]
    fn standby_and_off_are_silent() {
        for power in [PowerState::Standby, PowerState::Off] {
            let controls = SampleControls {
                power,
                ..SampleControls::default()
            };
            let mut engine = prepared_engine(&controls);
            let peak = peak_for(&mut engine, &controls, 1.0, 220.0);
            assert!(peak < 1.0e-6, "{power:?} leaked {peak}");
        }
    }

    #[test]
    fn lamp_tracks_the_power_switch_and_the_rail() {
        let controls = SampleControls::default();
        let mut engine = prepared_engine(&controls);
        assert_eq!(engine.lamp_brightness(PowerState::Off), 0.0);
        assert!(engine.lamp_brightness(PowerState::Standby) > 0.0);

        // Idle: rail full, lamp bright.
        for _ in 0..(FS as usize) {
            engine.process_sample(0.0, &controls);
        }
        let idle = engine.lamp_brightness(PowerState::On);

        // Hard playing: the rail sags and the lamp dips.
        for n in 0..(FS * 0.2) as usize {
            let x = (std::f32::consts::TAU * 110.0 * n as f32 / FS).sin();
            engine.process_sample(x, &controls);
        }
        let loaded = engine.lamp_brightness(PowerState::On);
        assert!(loaded < idle, "lamp did not dip: {idle} -> {loaded}");
        assert!((0.0..=1.0).contains(&loaded));
    }

    #[test]
    fn warmup_ramps_the_rail_up_rather_than_thumping() {
        let controls = SampleControls::default();
        let mut engine = AmpEngine::new();
        let standby = SampleControls {
            power: PowerState::Standby,
            ..controls
        };
        assert!(engine.prepare(FS, &standby));
        assert_eq!(engine.b_plus(), 0.0);

        // A tenth of the way through the warm-up the rail must be partial.
        for _ in 0..(FS * WARMUP_SECONDS * 0.1) as usize {
            engine.process_sample(0.0, &controls);
        }
        let early = engine.b_plus();
        assert!(early > 0.0 && early < 480.0, "rail jumped to {early} V");

        for _ in 0..(FS * WARMUP_SECONDS) as usize {
            engine.process_sample(0.0, &controls);
        }
        assert!(
            engine.b_plus() > 470.0,
            "rail never came up: {} V",
            engine.b_plus()
        );
    }

    #[test]
    fn cabinet_bypass_changes_the_tone_but_not_the_latency() {
        let with_cab = SampleControls::default();
        let without_cab = SampleControls {
            cab_enabled: false,
            ..SampleControls::default()
        };
        let a = prepared_engine(&with_cab);
        let b = prepared_engine(&without_cab);
        assert_eq!(a.latency_samples(), b.latency_samples());

        let mut a = prepared_engine(&with_cab);
        let mut b = prepared_engine(&without_cab);
        let wet = peak_for(&mut a, &with_cab, 0.3, 6_000.0);
        let dry = peak_for(&mut b, &without_cab, 0.3, 6_000.0);
        assert!(
            wet < dry * 0.6,
            "cabinet did not roll off 6 kHz: {dry} -> {wet}"
        );
    }

    #[test]
    fn tone_controls_are_audible_on_both_channels() {
        for channel in [Channel::Clean, Channel::Dirty] {
            let dark = SampleControls {
                channel,
                clean_treble: 0.0,
                dirty_treble: 0.0,
                ..SampleControls::default()
            };
            let bright = SampleControls {
                channel,
                clean_treble: 10.0,
                dirty_treble: 10.0,
                ..SampleControls::default()
            };
            let mut a = prepared_engine(&dark);
            let mut b = prepared_engine(&bright);
            let low = peak_for(&mut a, &dark, 0.2, 3_000.0);
            let high = peak_for(&mut b, &bright, 0.2, 3_000.0);
            assert!(
                high > low * 1.3,
                "{channel:?} treble did nothing: {low} -> {high}"
            );
        }
    }

    #[test]
    fn output_is_finite_for_pathological_input() {
        let controls = SampleControls::default();
        let mut engine = prepared_engine(&controls);
        for x in [1.0e9f32, -1.0e9, f32::MAX, f32::MIN, 0.0] {
            for _ in 0..256 {
                let y = engine.process_sample(x, &controls);
                assert!(y.is_finite(), "input {x} produced {y}");
            }
        }
        // And it recovers to silence afterwards.
        for _ in 0..(FS as usize) {
            engine.process_sample(0.0, &controls);
        }
        let mut peak = 0.0f32;
        for _ in 0..1_000 {
            peak = peak.max(engine.process_sample(0.0, &controls).abs());
        }
        assert!(peak < 1.0e-3, "engine did not settle: {peak}");
    }

    #[test]
    fn unprepared_engine_is_silent_rather_than_panicking() {
        let controls = SampleControls::default();
        let mut engine = AmpEngine::new();
        for _ in 0..128 {
            assert_eq!(engine.process_sample(1.0, &controls), 0.0);
        }
    }
}
