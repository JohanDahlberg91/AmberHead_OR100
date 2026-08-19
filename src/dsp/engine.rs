//! Complete OR100 signal chain.
//!
//! Implements the block diagram in specification section 2:
//!
//! ```text
//! In -> [8x Up] -> [Channel Switch]
//!         Clean: V1 -> 2-band EQ -> Clean Volume
//!         Dirty: V2 -> Gain -> V3 -> V4 (Boost) -> V5 -> 3-band EQ -> Vol
//!       -> [Global Boost +3 dB] -> [Driver V5] <- [Global NFB]
//!       -> [LTP PI] -> [EL34 Push-Pull] <- [B+ Sag]
//!       -> [8x Down] -> [Output Transformer] -> [Partitioned FFT Cab IR] -> Out
//! ```
//!
//! The driver stage and the feedback loop around it are shared by both
//! channels, as they are in the amplifier: the factory schematic has one
//! preamp, and everything the channel switch selects arrives at the same
//! 390 kΩ driver valve before the inverter.
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
//!
//! All four dirty-channel triodes are always in circuit. The gain pot is a
//! divider on V2's plate, so the dial's travel is spent driving the cascade
//! progressively harder rather than switching stages in and out. Nothing in
//! the chain clamps a plate node: a triode's swing is bounded by its own rail,
//! which is what lets the top of the dial keep adding drive.
//!
//! The cascade is what makes this a high-gain channel rather than a crunch
//! channel. What separates the two is not how hard the amplifier clips a
//! struck note — two stages manage that — but how far into the decay it stays
//! clipped. Each stage the gain pot drives multiplies the level reaching the
//! next, so the dirty channel's three cascaded stages plus the shared driver
//! hold saturation some 25 dB further down than a two-stage path does: the
//! channel sustains instead of cleaning up as a note fades.
//!
//! # Relationship to the factory schematic
//!
//! Every component value here is read off the Orange OR 100 schematic, print
//! HH A03057: 220 kΩ plate loads on 2.4 kΩ bypassed cathodes, a 390 kΩ driver
//! on 1 kΩ, 22 nF coupling throughout, the 100 kΩ/330 pF/22 nF/2.2 nF/27 kΩ
//! tone network, and the global feedback loop. Three things are deliberately
//! not the schematic's, because the modern reissue this plugin models is not
//! that amplifier: it has two channels where the original has one, a third
//! cascade stage in the dirty channel where the original has none, and a
//! long-tailed-pair inverter where the original uses a cathodyne. The pads
//! between stages are named and derived from real dividers; the only free
//! constants are the two makeup terms and [`DRIVER_TO_PI`], which exists
//! solely to undo the voltage gain a cathodyne would not have contributed.

use super::cabinet::Cabinet;
use super::denormal::{flush, sanitize};
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
/// range, chosen so that a full-scale input on the default patch (dirty
/// channel, gain 6.5, volume 5.0, 100 W) peaks at -10.1 dBFS, and the loudest
/// setting the front panel can reach — gain and volume at 10 with both boosts
/// engaged — lands at -1.3 dBFS rather than clipping the host bus.
pub const OUTPUT_CALIBRATION: f32 = 0.24;

/// Divider between V1's plate and the clean channel's tone stack.
const CLEAN_STACK_DRIVE: f32 = 0.35;
/// Makeup applied after the clean tone stack's insertion loss, so the clean
/// channel can still reach breakup at high volume settings.
///
/// The clean channel has no cascade behind it, so this constant alone decides
/// how hard volume 10 drives the shared driver valve. At 0.85 the channel runs
/// from -23.5 dBFS and 1.0 % distortion at volume 3 to -4.8 dBFS and 44 % at
/// volume 10: recognisably clean to about 6, an edge-of-breakup range above
/// it, and monotonic all the way up.
const CLEAN_MAKEUP: f32 = 0.85;

/// Loading of V2's plate by the 1 MΩ gain pot at full rotation.
///
/// The pot sits directly on V2's coupling capacitor and is the whole load that
/// plate sees, so at maximum the wiper carries `Rpot / (Rpot + Zplate)` of the
/// plate signal: `1M / (1M + 48k) = 0.954`, where `Zplate = Ra || rp =
/// 220k || 62k`. A cascade's gain pot is a divider, not an attenuator — at 10
/// it hands the next grid essentially the entire plate swing.
///
/// The schematic wires its one volume pot in exactly this position, between
/// the first and second triode, which is why the original's `Volume` is a
/// gain control and not a master.
const V2_TO_GAIN_POT: f32 = 0.954;

/// Divider between V3's plate and V4's grid with the gain boost off.
///
/// 22 nF into a 390 kΩ + 82 kΩ series leg with a 100 kΩ shunt to ground.
/// Against the 48 kΩ plate impedance and V4's own 1 MΩ grid leak in parallel
/// with the shunt (`100k || 1M = 90.9k`) that is
/// `90.9k / (48k + 472k + 90.9k) = 0.149`.
///
/// A high-gain cascade pads only lightly between stages. The pad's job is to
/// stop the *loudest* notes from blocking the next grid solid, not to decide
/// whether that grid clips at all — every note above a whisper is meant to
/// clip it. Where it sits decides how far into a note's decay the cascade
/// stays saturated: at this value the channel is clean at 1 on the dial,
/// breaking up by 2, and at 10 holds saturation well below -30 dBFS of input.
const V3_TO_V4: f32 = 0.149;

/// The same divider with the gain boost engaged.
///
/// The switch shorts the 390 kΩ resistor, leaving the 82 kΩ grid stopper:
/// `90.9k / (48k + 82k + 90.9k) = 0.412`, a +8.8 dB step. The boost changes
/// how hard V4 is driven, not whether it is in circuit — every stage conducts
/// in either switch position, which is what keeps the channel's voicing
/// constant across the switch.
const V3_TO_V4_BOOSTED: f32 = 0.412;

/// Divider between V4's plate and V5's grid, the last pad in the cascade.
///
/// 22 nF into a 1 MΩ series resistor with a 68 kΩ shunt to ground:
/// `63.7k / (48k + 1M + 63.7k) = 0.057`, where `63.7k = 68k || 1M`. Heavier
/// than the pad ahead of V4, because V4 arrives already squared up: V5 is
/// there to compress and sustain what the cascade has produced, and blocking
/// its grid solid on every note would only smear the attack.
const V4_TO_V5: f32 = 0.057;

/// Divider between the last cascade plate and the dirty tone stack.
///
/// The stack hangs off V4's plate through the same 22 nF coupling capacitor;
/// its 100 kΩ slope resistor and 1 MΩ pots load that plate almost as lightly
/// as another grid would.
const DIRTY_STACK_DRIVE: f32 = 0.90;

/// Pad on the dirty channel between the volume pot and the driver's grid.
///
/// Three cascaded stages coupled at their real circuit ratios put tens of
/// volts of square wave into the stack, so this constant *attenuates*: it
/// stands in for the loading the volume pot's wiper and the driver's grid
/// network apply. The channel's drive comes from the cascade, not from here.
///
/// With the cascade's own pads it sets where on the dial saturation arrives —
/// 0.3 % distortion at 1, 17 % by 3, 81 % at 10 — and how far into a note's
/// decay it holds: at 10 the channel is still at 25 % with only -34 dBFS of
/// input, having compressed 26 dB of input range into 8 dB of output.
const DIRTY_MAKEUP: f32 = 0.030;

/// Divider between either channel's volume pot and the driver's grid.
///
/// 22 nF into a 68 kΩ grid stopper and the driver's 1 MΩ grid leak, fed from
/// the volume pot's wiper: `1M / (1M + 68k) = 0.94` at the top of the dial.
const VOLUME_TO_DRIVER: f32 = 0.94;

/// Pad between the driver's plate and the phase inverter's grid.
///
/// This is the one constant with no counterpart on the schematic, and it is
/// there because the inverter does not match either. The factory circuit ends
/// in a cathodyne, which has no voltage gain: its driver has to supply the
/// whole 30-odd volts an EL34 grid needs, which is exactly why that stage
/// carries a 390 kΩ load and clips as early as it does. The long-tailed pair
/// this plugin uses contributes roughly 30x of its own, so the driver's output
/// is padded by about that much before it reaches the inverter's grid. Without
/// it the amplifier would be an oscillator's worth of gain too hot, and the
/// inverter would be blocking on every note.
const DRIVER_TO_PI: f32 = 0.033;

/// Fraction of the output transformer's secondary voltage fed back into the
/// driver, from the schematic's global feedback loop.
///
/// The loop is a 24 kΩ resistor, shunted by 1 nF, into a 150 Ω resistor to
/// ground: `beta = 150 / (24k + 150) = 0.0062`. The secondary swings about
/// 40 V peak at full output on the 100 W setting, against a power stage whose
/// normalized output reaches 2.0, so the volts arriving back at the driver are
/// `0.0062 * 40/2 = 0.124` per unit of power-stage output.
///
/// The 1 nF across the 24 kΩ lifts the feedback above roughly 6.6 kHz, which
/// is a presence rolloff rather than a flat loop; only the flat term is
/// modelled here.
const NFB_FACTOR: f32 = 0.124;

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
    /// Shorts the 390 kOhm resistor in front of V4's grid for roughly 9 dB
    /// more drive into the back half of the cascade.
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
    dirty_v5: Triode,
    dirty_stack: ToneStack,

    /// Shared driver valve between the channel switch and the inverter.
    driver: Triode,
    phase_inverter: PhaseInverter,
    power_amp: PowerAmp,
    transformer: OutputTransformer,
    cabinet: Cabinet,

    channel_ramp: SwitchRamp,
    gain_boost_ramp: SwitchRamp,
    global_boost_ramp: SwitchRamp,

    /// Feedback voltage from the previous sample's power-stage output.
    nfb: f32,
    /// Depth of the global feedback loop, [`NFB_FACTOR`] in normal operation.
    nfb_factor: f32,

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
            dirty_v5: Triode::new(StageCircuit::cascade_stage()),
            dirty_stack: ToneStack::new(ToneStackCircuit::or100_dirty()),
            driver: Triode::new(StageCircuit::driver_stage()),
            phase_inverter: PhaseInverter::new(),
            power_amp: PowerAmp::new(),
            transformer: OutputTransformer::new(),
            cabinet: Cabinet::new(),
            nfb: 0.0,
            nfb_factor: NFB_FACTOR,
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
        self.dirty_v5.prepare(oversampled_rate);
        self.driver.prepare(oversampled_rate);

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
        self.dirty_v5.reset();
        self.dirty_stack.reset();
        self.driver.reset();
        self.nfb = 0.0;
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

    /// Swaps in a measured impulse response.
    ///
    /// Delegates to [`Cabinet::load_ir`], which copies into buffers that
    /// already exist and re-runs the partitioning transforms in place, so this
    /// allocates nothing and is safe to call between blocks on the audio
    /// thread. Returns `false` if the engine is unprepared or the taps are
    /// unusable, leaving the previous cabinet running.
    pub fn load_impulse_response(&mut self, taps: &[f32]) -> bool {
        self.prepared && self.cabinet.load_ir(taps)
    }

    /// Puts the synthesised 4x12 response back. Allocation-free, as above.
    pub fn restore_default_impulse_response(&mut self) -> bool {
        self.prepared && self.cabinet.restore_default_ir()
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

    /// Opens the global feedback loop, for tests that need an open-loop
    /// reference. The loop cannot be defeated from outside the crate.
    #[cfg(test)]
    fn set_feedback_depth(&mut self, depth: f32) {
        self.nfb_factor = depth;
        self.nfb = 0.0;
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
        // Gain boost shorts the pad in front of V4's grid; the ramp crossfades
        // the divider ratio rather than the audio, so the stage never leaves
        // the signal path and cannot click.
        let v4_drive = V3_TO_V4 + (V3_TO_V4_BOOSTED - V3_TO_V4) * boost_mix;

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
            dirty_v5,
            dirty_stack,
            driver,
            phase_inverter,
            power_amp,
            nfb,
            nfb_factor,
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

            // --- Dirty: V2 -> V3 -> V4 -> V5 -> 3-band EQ -> Volume ---------
            // Every stage is always in circuit. The gain pot sets how hard V2
            // hits the cascade, and the boost switch shorts the series resistor
            // in front of V4; the stack and the volume pot only ever attenuate
            // what the cascade has already done.
            let dirty = if channel_mix > 0.0 {
                let v2 = dirty_v2.process(sample);
                let v3 = dirty_v3.process(v2 * gain * V2_TO_GAIN_POT);
                let v4 = dirty_v4.process(v3 * v4_drive);
                let v5 = dirty_v5.process(v4 * V4_TO_V5);

                let shaped = dirty_stack.process(v5 * DIRTY_STACK_DRIVE);
                shaped * dirty_volume * DIRTY_MAKEUP
            } else {
                dirty_v2.process(0.0);
                dirty_v3.process(0.0);
                dirty_v4.process(0.0);
                dirty_v5.process(0.0);
                dirty_stack.process(0.0);
                0.0
            };

            let preamp = clean + (dirty - clean) * channel_mix;

            // --- Driver, global feedback, phase inverter, power stage -------
            // The schematic closes its feedback loop on the last small-signal
            // stage, so the loop is subtracted at the driver's grid. The
            // one-sample delay this costs is 2.6 µs at the oversampled rate,
            // three orders of magnitude inside the loop's own bandwidth.
            let driven = preamp * global_gain * VOLUME_TO_DRIVER - *nfb;
            let driver_out = driver.process(driven);
            let differential = phase_inverter.process(driver_out * DRIVER_TO_PI);
            let power_output = power_amp.process(differential);
            *nfb = flush(*nfb_factor * power_output);
            power_output
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
        // The cabinet is bypassed for every distortion measurement. A speaker
        // is linear: it cannot add or remove a single harmonic. What it does
        // do is tilt the spectrum by tens of decibels, and since THD is a
        // ratio of harmonic energy to the fundamental, that tilt moves the
        // number without the amplifier's behaviour changing at all — fitting
        // the cabinet to a measured 4x12 shifted these readings by a factor of
        // three while the preamp was untouched. Bypassing it measures the
        // amplifier, which is what these tests are about.
        let controls = &SampleControls {
            cab_enabled: false,
            ..*controls
        };
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
    fn gain_control_spans_clean_to_fully_saturated() {
        // The four cascade stages are all in circuit, so the dial's travel is
        // spent driving them progressively harder: a trace of grit at 1, well
        // into clipping by 3, fully saturated at the top.
        //
        // Regression guard. When each triode's plate was clamped at ±32 V, V3
        // pinned at the clamp by 4 on the dial and everything above it was
        // bit-identical — the knob was dead over its whole upper half.
        let thd = |gain: f32| {
            engine_thd_percent(
                &SampleControls {
                    dirty_gain: gain,
                    ..SampleControls::default()
                },
                0.2,
            )
        };

        let (bottom, middle, upper, top) = (thd(1.0), thd(3.0), thd(5.0), thd(10.0));
        assert!(bottom < 2.0, "gain 1 is not clean: {bottom}%");
        assert!(middle > 10.0, "gain 3 has not broken up: {middle}%");
        assert!(upper > 25.0, "gain 5 only reached {upper}%");
        // Past the point where the cascade squares up, more drive buys
        // compression and sustain rather than a higher harmonic ratio — the
        // reading is not even monotonic up here, because grid blocking
        // reshapes the waveform — so the top of the dial is checked against an
        // absolute floor rather than against the rung below it.
        assert!(top > 35.0, "the top of the dial only reached {top}% THD");
    }

    #[test]
    fn saturation_survives_a_note_decaying_by_twenty_six_decibels() {
        // What separates this amplifier from a 30 W crunch combo is not how
        // hard it clips a struck note — any three-stage channel manages that —
        // but how far into the decay it stays clipped. Four cascaded stages
        // hold saturation some 25 dB further down than three do.
        //
        // The chain is checked at gain 10 against inputs 26 dB apart, -8 and
        // -34 dBFS. Both must stay heavily distorted, and the output must
        // compress hard: a fully saturated cascade barely changes level as the
        // note fades, which is the sustain the amplifier is bought for.
        let controls = SampleControls {
            dirty_gain: 10.0,
            ..SampleControls::default()
        };

        let struck = engine_thd_percent(&controls, 0.4);
        let decayed = engine_thd_percent(&controls, 0.02);
        assert!(struck > 40.0, "a struck note only reached {struck}%");
        assert!(
            decayed > 20.0,
            "the channel cleaned up as the note decayed: {decayed}%"
        );

        let bare = SampleControls {
            cab_enabled: false,
            ..controls
        };
        let mut loud_engine = prepared_engine(&bare);
        let mut quiet_engine = prepared_engine(&bare);
        let loud = peak_for(&mut loud_engine, &bare, 0.4, 220.0);
        let quiet = peak_for(&mut quiet_engine, &bare, 0.02, 220.0);
        let compression = 20.0 * (loud / quiet.max(1.0e-9)).log10();
        assert!(
            compression < 12.0,
            "26 dB of input became {compression} dB of output: not compressing"
        );
    }

    #[test]
    fn global_feedback_reduces_gain_rather_than_adding_it() {
        // The schematic's loop is 24 kOhm into 150 Ohm off the transformer
        // secondary, subtracted at the driver's grid. Sign errors in a
        // feedback path are silent until the amplifier oscillates, so this
        // pins the polarity: with the loop closed, small-signal gain must be
        // *lower* than the same chain running open, and the amplifier must
        // stay bounded when driven hard.
        let controls = SampleControls {
            dirty_gain: 3.0,
            dirty_volume: 3.0,
            ..SampleControls::default()
        };

        let mut closed = prepared_engine(&controls);
        let with_loop = peak_for(&mut closed, &controls, 0.02, 220.0);

        let mut open = prepared_engine(&controls);
        open.set_feedback_depth(0.0);
        let without_loop = peak_for(&mut open, &controls, 0.02, 220.0);

        assert!(
            with_loop < without_loop,
            "feedback raised the gain: {without_loop} open -> {with_loop} closed"
        );
        let depth = 20.0 * (without_loop / with_loop.max(1.0e-9)).log10();
        assert!(
            (0.5..=6.0).contains(&depth),
            "feedback depth measured {depth} dB, not the ~1.5 dB the divider is worth"
        );
    }

    #[test]
    fn gain_boost_is_a_drive_step_not_a_stage_swap() {
        // The boost shorts the 390 kΩ resistor in front of V4's grid, so it is
        // worth a fixed ~9 dB of extra drive rather than switching a whole
        // stage in and out.
        //
        // Measured at the bottom of the gain dial, which is the only place the
        // step shows up as level: anywhere above it the cascade is already
        // clipping, and 9 dB more drive into a clipping stage buys distortion
        // and compression instead. See
        // [`Self::gain_boost_adds_saturation_when_the_cascade_is_clipping`].
        let plain = SampleControls {
            dirty_gain: 1.0,
            ..SampleControls::default()
        };
        let boosted = SampleControls {
            gain_boost: true,
            ..plain
        };
        let mut a = prepared_engine(&plain);
        let mut b = prepared_engine(&boosted);
        let quiet = peak_for(&mut a, &plain, 0.2, 220.0);
        let loud = peak_for(&mut b, &boosted, 0.2, 220.0);
        let delta = 20.0 * (loud / quiet.max(1.0e-9)).log10();
        assert!(
            (7.0..=12.0).contains(&delta),
            "gain boost stepped by {delta} dB, not the ~9 dB the pad is worth"
        );
    }

    #[test]
    fn a_fully_saturated_cascade_still_does_not_alias() {
        // `CLAUDE.md` §5 aliasing test, applied to the whole amplifier at the
        // setting that generates the most harmonic energy: four clipped stages
        // plus a slammed power stage. The cabinet is bypassed so its 4.8 kHz
        // rolloff cannot flatter the measurement.
        //
        // A 2.5 kHz fundamental puts its harmonics on multiples of 2.5 kHz and
        // its folded images on 500 Hz, 2 kHz, 4.5 kHz, 7 kHz and 9.5 kHz —
        // none of which is a harmonic, so anything measured there is aliasing.
        let controls = SampleControls {
            dirty_gain: 10.0,
            gain_boost: true,
            dirty_volume: 8.0,
            cab_enabled: false,
            ..SampleControls::default()
        };
        let mut engine = prepared_engine(&controls);

        let settle = (FS * 0.3) as usize;
        for n in 0..settle {
            let x = 0.5 * (std::f32::consts::TAU * 2_500.0 * n as f32 / FS).sin();
            engine.process_sample(x, &controls);
        }
        let count = 16_384usize;
        let mut samples = Vec::with_capacity(count);
        for n in 0..count {
            let x = 0.5 * (std::f32::consts::TAU * 2_500.0 * (settle + n) as f32 / FS).sin();
            samples.push(engine.process_sample(x, &controls) as f64);
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

        let fundamental = magnitude(2_500.0).max(1.0e-12);
        for alias in [500.0, 2_000.0, 4_500.0, 7_000.0, 9_500.0] {
            let level = 20.0 * (magnitude(alias) / fundamental).log10();
            assert!(level < -60.0, "alias at {alias} Hz was {level} dB");
        }
    }

    #[test]
    fn the_cascade_saturates_without_the_gain_boost() {
        // All four stages are in circuit in both switch positions, so the
        // channel reaches heavy saturation on the gain control alone. The
        // boost is a hotter setting, not the difference between a crunch amp
        // and a high-gain one.
        let controls = SampleControls {
            dirty_gain: 10.0,
            gain_boost: false,
            ..SampleControls::default()
        };
        let thd = engine_thd_percent(&controls, 0.2);
        assert!(thd > 35.0, "unboosted cascade only reached {thd}%");
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
        assert!(low < 5.0, "gain 1 already distorting at {low}%");
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
    fn gain_control_drives_harder_until_the_amplifier_compresses() {
        // Cabinet bypassed for the same reason as `engine_thd_percent`: the
        // speaker's spectral tilt moves the level of a distorted waveform
        // around by more than the gain control does near the top of its
        // travel, so with it in circuit this measures the cabinet.
        //
        // The dial raises the output through its useful range and then stops,
        // which is not a defect but the behaviour being modelled: once the
        // cascade is fully saturated the ceiling is set by the amplifier, and
        // further drive goes into grid blocking, which if anything costs a
        // little level. A captured Orange shows the same thing — its output
        // moves under half a decibel across a 26 dB change of input. So the
        // bottom of the dial must climb, and the top must hold station rather
        // than collapse.
        let level = |gain: f32| -> f32 {
            let controls = SampleControls {
                dirty_gain: gain,
                cab_enabled: false,
                ..SampleControls::default()
            };
            let mut engine = prepared_engine(&controls);
            let settle = (FS * 0.3) as usize;
            for n in 0..settle {
                let x = 0.3 * (std::f32::consts::TAU * 220.0 * n as f32 / FS).sin();
                engine.process_sample(x, &controls);
            }
            let count = (FS * 0.2) as usize;
            let mut sum = 0.0f64;
            for n in 0..count {
                let x = 0.3 * (std::f32::consts::TAU * 220.0 * (settle + n) as f32 / FS).sin();
                let y = engine.process_sample(x, &controls) as f64;
                sum += y * y;
            }
            (sum / count as f64).sqrt() as f32
        };

        let (one, two, three) = (level(1.0), level(2.0), level(3.0));
        assert!(
            two > one * 2.0,
            "gain 1 -> 2 gained nothing: {one} -> {two}"
        );
        assert!(three > two, "gain 2 -> 3 gained nothing: {two} -> {three}");

        for gain in [5.0f32, 8.0, 10.0] {
            let held = level(gain);
            let drop_db = 20.0 * (three / held).log10();
            assert!(
                drop_db < 2.5,
                "gain {gain} fell {drop_db} dB below gain 3: the top of the dial is collapsing"
            );
        }
    }

    #[test]
    fn gain_boost_adds_saturation_when_the_cascade_is_clipping() {
        // On the default patch the cascade is already well into clipping, so
        // the boost's extra drive comes out as harmonic content rather than as
        // level — the same thing a real amplifier does once its preamp is
        // squared up.
        let plain = SampleControls::default();
        let boosted = SampleControls {
            gain_boost: true,
            ..SampleControls::default()
        };
        let quiet = engine_thd_percent(&plain, 0.2);
        let loud = engine_thd_percent(&boosted, 0.2);
        assert!(
            loud > quiet * 1.3,
            "gain boost did nothing: {quiet}% -> {loud}%"
        );
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
