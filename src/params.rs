//! Parameter schema.
//!
//! Implements the table in specification section 3 verbatim: fourteen
//! automatable parameters, with exponential ranges and 15/20 ms smoothing on
//! the continuous controls and zero smoothing on the discrete switches
//! (`CLAUDE.md` §3).
//!
//! The DSP core in [`crate::dsp`] knows nothing about these types. This module
//! is the only place where `nih-plug` parameter types and DSP enumerations
//! meet, which keeps the amplifier model host-agnostic and unit-testable.

use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;

use crate::dsp::engine::{Channel, PowerState, SampleControls};
use crate::dsp::power::PowerMode;

/// Smoothing time for the volume and gain controls, in ms
/// (specification section 3).
const VOLUME_SMOOTHING_MS: f32 = 15.0;
/// Smoothing time for the tone controls, in ms (specification section 3).
const TONE_SMOOTHING_MS: f32 = 20.0;

/// Skew factor for the volume and gain ranges.
///
/// `FloatRange::Skewed` unnormalizes as `normalized^(1/factor)`, so a factor
/// below one expands the lower half of the range. 0.65 gives roughly twice the
/// automation resolution below 5.0 as above it, which is where a guitar
/// amplifier's audible changes are concentrated, while keeping the curve gentle
/// enough that a host's linear automation ramp still sounds linear.
const VOLUME_SKEW: f32 = 0.65;

/// Preamp channel selection.
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelParam {
    /// Single-stage clean channel with the 2-band tone stack.
    #[id = "clean"]
    #[name = "Clean"]
    Clean,
    /// Cascaded gain channel with the 3-band tone stack.
    #[id = "dirty"]
    #[name = "Dirty"]
    Dirty,
}

impl From<ChannelParam> for Channel {
    fn from(value: ChannelParam) -> Self {
        match value {
            ChannelParam::Clean => Channel::Clean,
            ChannelParam::Dirty => Channel::Dirty,
        }
    }
}

/// Power / standby switch position.
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerParam {
    /// Mains off.
    #[id = "off"]
    #[name = "Off"]
    Off,
    /// Heaters on, `B+` disconnected.
    #[id = "standby"]
    #[name = "Standby"]
    Standby,
    /// Fully operational.
    #[id = "on"]
    #[name = "On"]
    On,
}

impl From<PowerParam> for PowerState {
    fn from(value: PowerParam) -> Self {
        match value {
            PowerParam::Off => PowerState::Off,
            PowerParam::Standby => PowerState::Standby,
            PowerParam::On => PowerState::On,
        }
    }
}

/// Output wattage mode, per the switching matrix in specification section 2.C.
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputPowerParam {
    /// 4 tubes, 480 V rail.
    #[id = "100w"]
    #[name = "100 W"]
    Watt100,
    /// 4 tubes, 340 V rail.
    #[id = "70w"]
    #[name = "70 W"]
    Watt70,
    /// 2 tubes, 480 V rail.
    #[id = "50w"]
    #[name = "50 W"]
    Watt50,
    /// 2 tubes, 340 V rail.
    #[id = "30w"]
    #[name = "30 W"]
    Watt30,
}

impl From<OutputPowerParam> for PowerMode {
    fn from(value: OutputPowerParam) -> Self {
        match value {
            OutputPowerParam::Watt100 => PowerMode::Watt100,
            OutputPowerParam::Watt70 => PowerMode::Watt70,
            OutputPowerParam::Watt50 => PowerMode::Watt50,
            OutputPowerParam::Watt30 => PowerMode::Watt30,
        }
    }
}

/// The plugin's automatable parameters.
#[derive(Params)]
pub struct Or100Params {
    /// Editor size, persisted with the rest of the plugin state so a resized
    /// window survives a session reload.
    #[persist = "editor-state"]
    pub editor_state: Arc<ViziaState>,

    #[id = "channel"]
    pub channel: EnumParam<ChannelParam>,

    #[id = "clean_volume"]
    pub clean_volume: FloatParam,
    #[id = "clean_bass"]
    pub clean_bass: FloatParam,
    #[id = "clean_treble"]
    pub clean_treble: FloatParam,

    #[id = "dirty_gain"]
    pub dirty_gain: FloatParam,
    #[id = "dirty_bass"]
    pub dirty_bass: FloatParam,
    #[id = "dirty_middle"]
    pub dirty_middle: FloatParam,
    #[id = "dirty_treble"]
    pub dirty_treble: FloatParam,
    #[id = "dirty_volume"]
    pub dirty_volume: FloatParam,

    #[id = "gain_boost"]
    pub gain_boost: BoolParam,
    #[id = "global_boost"]
    pub global_boost: BoolParam,

    #[id = "power_switch"]
    pub power_switch: EnumParam<PowerParam>,
    #[id = "output_power"]
    pub output_power: EnumParam<OutputPowerParam>,
    #[id = "cab_enabled"]
    pub cab_enabled: BoolParam,
}

/// Builds a 0.0..=10.0 front-panel control with an exponential taper.
fn exponential_knob(name: &str, default: f32) -> FloatParam {
    FloatParam::new(
        name,
        default,
        FloatRange::Skewed {
            min: 0.0,
            max: 10.0,
            factor: VOLUME_SKEW,
        },
    )
    .with_smoother(SmoothingStyle::Exponential(VOLUME_SMOOTHING_MS))
    .with_step_size(0.01)
}

/// Builds a 0.0..=10.0 front-panel control with a linear taper.
fn linear_knob(name: &str, default: f32) -> FloatParam {
    FloatParam::new(
        name,
        default,
        FloatRange::Linear {
            min: 0.0,
            max: 10.0,
        },
    )
    .with_smoother(SmoothingStyle::Exponential(TONE_SMOOTHING_MS))
    .with_step_size(0.01)
}

impl Default for Or100Params {
    fn default() -> Self {
        Self {
            editor_state: crate::gui::default_state(),

            channel: EnumParam::new("Channel Select", ChannelParam::Dirty),

            clean_volume: exponential_knob("Clean Volume", 5.0),
            clean_bass: linear_knob("Clean Bass", 5.0),
            clean_treble: linear_knob("Clean Treble", 5.0),

            dirty_gain: exponential_knob("Dirty Gain", 6.5),
            dirty_bass: linear_knob("Dirty Bass", 5.0),
            dirty_middle: linear_knob("Dirty Middle", 5.0),
            dirty_treble: linear_knob("Dirty Treble", 5.0),
            dirty_volume: exponential_knob("Dirty Volume", 5.0),

            // Discrete switches carry no smoother. `global_boost`'s 10 ms ramp
            // from specification section 3 lives in the DSP core instead, as a
            // `SwitchRamp` crossfade, because `BoolParam` has no smoother of
            // its own (`CLAUDE.md` §3).
            gain_boost: BoolParam::new("Gain Boost", false),
            global_boost: BoolParam::new("Global Boost", false),

            power_switch: EnumParam::new("Power / Standby", PowerParam::On),
            output_power: EnumParam::new("Power Mode", OutputPowerParam::Watt100),
            cab_enabled: BoolParam::new("Cabinet Emulation", true),
        }
    }
}

impl Or100Params {
    /// Reads the discrete switch positions, which are not smoothed.
    ///
    /// Called once per block; the continuous fields of the returned struct are
    /// placeholders that [`Self::fill_smoothed`] overwrites per sample.
    pub fn block_controls(&self) -> SampleControls {
        SampleControls {
            channel: self.channel.value().into(),
            clean_volume: 0.0,
            clean_bass: 0.0,
            clean_treble: 0.0,
            dirty_gain: 0.0,
            dirty_bass: 0.0,
            dirty_middle: 0.0,
            dirty_treble: 0.0,
            dirty_volume: 0.0,
            gain_boost: self.gain_boost.value(),
            global_boost: self.global_boost.value(),
            power: self.power_switch.value().into(),
            output_power: self.output_power.value().into(),
            cab_enabled: self.cab_enabled.value(),
        }
    }

    /// Advances every smoother by one sample and writes the results into
    /// `controls`.
    ///
    /// All eight smoothers are stepped unconditionally, even for the channel
    /// that is not selected: skipping them would make a smoother jump when the
    /// channel is switched back, and would make the amplifier's output depend
    /// on how long ago a control was moved.
    #[inline]
    pub fn fill_smoothed(&self, controls: &mut SampleControls) {
        controls.clean_volume = self.clean_volume.smoothed.next();
        controls.clean_bass = self.clean_bass.smoothed.next();
        controls.clean_treble = self.clean_treble.smoothed.next();
        controls.dirty_gain = self.dirty_gain.smoothed.next();
        controls.dirty_bass = self.dirty_bass.smoothed.next();
        controls.dirty_middle = self.dirty_middle.smoothed.next();
        controls.dirty_treble = self.dirty_treble.smoothed.next();
        controls.dirty_volume = self.dirty_volume.smoothed.next();
    }

    /// Snapshot of every control with the smoothers at their target values.
    ///
    /// Used by `Plugin::initialize` to build the DSP state around the settings
    /// the plugin is about to start with, so the first block does not have to
    /// ramp in from a default patch.
    pub fn settled_controls(&self) -> SampleControls {
        SampleControls {
            clean_volume: self.clean_volume.value(),
            clean_bass: self.clean_bass.value(),
            clean_treble: self.clean_treble.value(),
            dirty_gain: self.dirty_gain.value(),
            dirty_bass: self.dirty_bass.value(),
            dirty_middle: self.dirty_middle.value(),
            dirty_treble: self.dirty_treble.value(),
            dirty_volume: self.dirty_volume.value(),
            ..self.block_controls()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_specification_table() {
        let params = Or100Params::default();
        assert_eq!(params.channel.value(), ChannelParam::Dirty);
        assert_eq!(params.clean_volume.value(), 5.0);
        assert_eq!(params.clean_bass.value(), 5.0);
        assert_eq!(params.clean_treble.value(), 5.0);
        assert_eq!(params.dirty_gain.value(), 6.5);
        assert_eq!(params.dirty_bass.value(), 5.0);
        assert_eq!(params.dirty_middle.value(), 5.0);
        assert_eq!(params.dirty_treble.value(), 5.0);
        assert_eq!(params.dirty_volume.value(), 5.0);
        assert!(!params.gain_boost.value());
        assert!(!params.global_boost.value());
        assert_eq!(params.power_switch.value(), PowerParam::On);
        assert_eq!(params.output_power.value(), OutputPowerParam::Watt100);
        assert!(params.cab_enabled.value());
    }

    #[test]
    fn the_schema_declares_exactly_fourteen_parameters() {
        let params = Or100Params::default();
        // `editor_state` is persisted state, not an automatable parameter, so
        // it must not appear here.
        assert_eq!(params.param_map().len(), 14);
    }

    #[test]
    fn parameter_ids_are_unique_and_match_the_specification() {
        let params = Or100Params::default();
        let param_map = params.param_map();
        let mut ids: Vec<&str> = param_map.iter().map(|(id, _, _)| id.as_str()).collect();
        ids.sort_unstable();
        let mut expected = vec![
            "channel",
            "clean_volume",
            "clean_bass",
            "clean_treble",
            "dirty_gain",
            "dirty_bass",
            "dirty_middle",
            "dirty_treble",
            "dirty_volume",
            "gain_boost",
            "global_boost",
            "power_switch",
            "output_power",
            "cab_enabled",
        ];
        expected.sort_unstable();
        assert_eq!(ids, expected);
    }

    #[test]
    fn every_continuous_control_spans_zero_to_ten() {
        let params = Or100Params::default();
        for param in [
            &params.clean_volume,
            &params.clean_bass,
            &params.clean_treble,
            &params.dirty_gain,
            &params.dirty_bass,
            &params.dirty_middle,
            &params.dirty_treble,
            &params.dirty_volume,
        ] {
            assert_eq!(param.preview_plain(0.0), 0.0);
            assert!((param.preview_plain(1.0) - 10.0).abs() < 1.0e-4);
            // Monotonic across the whole range.
            let mut previous = -1.0;
            for step in 0..=100 {
                let value = param.preview_plain(step as f32 / 100.0);
                assert!(value >= previous, "range is not monotonic");
                previous = value;
            }
        }
    }

    #[test]
    fn exponential_controls_expand_the_lower_half() {
        let params = Or100Params::default();
        // Half rotation must land below the midpoint on an exponential taper,
        // and the linear controls must land exactly on it.
        assert!(params.dirty_gain.preview_plain(0.5) < 4.5);
        assert!((params.dirty_bass.preview_plain(0.5) - 5.0).abs() < 1.0e-4);
    }

    #[test]
    fn discrete_switches_are_stepped_not_continuous() {
        let params = Or100Params::default();
        // `CLAUDE.md` §3: the switches must be discrete. A stepped parameter
        // reports a step count; a continuous one reports `None`.
        assert_eq!(params.channel.step_count(), Some(1));
        assert_eq!(params.gain_boost.step_count(), Some(1));
        assert_eq!(params.global_boost.step_count(), Some(1));
        assert_eq!(params.cab_enabled.step_count(), Some(1));
        assert_eq!(params.power_switch.step_count(), Some(2));
        assert_eq!(params.output_power.step_count(), Some(3));
    }

    #[test]
    fn enum_switches_round_trip_every_variant() {
        let params = Or100Params::default();
        // Each detent the lever switch can select must land back on a distinct
        // variant, which is what makes the four-position wattage lever work.
        let last = 3.0f32;
        let mut seen = Vec::new();
        for index in 0..=3 {
            let normalized = index as f32 / last;
            let rendered = params
                .output_power
                .normalized_value_to_string(normalized, false);
            assert!(!rendered.is_empty());
            assert!(!seen.contains(&rendered), "duplicate detent: {rendered}");
            seen.push(rendered);
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn dsp_conversions_cover_every_variant() {
        assert_eq!(Channel::from(ChannelParam::Clean), Channel::Clean);
        assert_eq!(Channel::from(ChannelParam::Dirty), Channel::Dirty);
        assert_eq!(PowerState::from(PowerParam::Off), PowerState::Off);
        assert_eq!(PowerState::from(PowerParam::Standby), PowerState::Standby);
        assert_eq!(PowerState::from(PowerParam::On), PowerState::On);
        assert_eq!(
            PowerMode::from(OutputPowerParam::Watt100),
            PowerMode::Watt100
        );
        assert_eq!(PowerMode::from(OutputPowerParam::Watt70), PowerMode::Watt70);
        assert_eq!(PowerMode::from(OutputPowerParam::Watt50), PowerMode::Watt50);
        assert_eq!(PowerMode::from(OutputPowerParam::Watt30), PowerMode::Watt30);
    }

    #[test]
    fn settled_controls_report_the_current_parameter_values() {
        let params = Or100Params::default();
        let controls = params.settled_controls();
        assert_eq!(controls.dirty_gain, 6.5);
        assert_eq!(controls.channel, Channel::Dirty);
        assert_eq!(controls.output_power, PowerMode::Watt100);
        assert!(controls.cab_enabled);
    }

    #[test]
    fn smoothers_converge_on_their_targets() {
        let params = Or100Params::default();
        let sample_rate = 48_000.0;
        // 15 ms of smoothing at 48 kHz.
        params.dirty_gain.smoothed.set_target(sample_rate, 9.0);
        let mut controls = params.block_controls();
        for _ in 0..(sample_rate * 0.1) as usize {
            params.fill_smoothed(&mut controls);
        }
        assert!(
            (controls.dirty_gain - 9.0).abs() < 0.01,
            "smoother reached {}",
            controls.dirty_gain
        );
    }
}
