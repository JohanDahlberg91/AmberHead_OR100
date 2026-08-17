//! AmberHead OR100 — a virtual analog recreation of the Orange OR100 Modern
//! Reissue tube amplifier, targeting VST3, CLAP and standalone.
//!
//! Module layout:
//!
//! * [`dsp`] — the amplifier model. Host-agnostic, unit-testable, and the only
//!   code that runs on the audio thread.
//! * [`params`] — the fourteen-parameter schema from specification section 3.
//! * [`gui`] — the `vizia` faceplate. Never references [`dsp`] types.
//! * [`shared`] — the one lock-free value that crosses the audio/UI boundary.

#![deny(unsafe_op_in_unsafe_fn)]

use std::sync::Arc;

use nih_plug::prelude::*;

pub mod dsp;
pub mod gui;
pub mod params;
pub mod shared;

use dsp::denormal::DenormalGuard;
use dsp::engine::{AmpEngine, SampleControls};
use params::Or100Params;
use shared::AtomicLevel;

/// The plugin.
pub struct AmberHeadOr100 {
    params: Arc<Or100Params>,
    /// The amplifier. Owned by the audio thread; the editor never sees it.
    engine: Box<AmpEngine>,
    /// Jewel lamp brightness, published for the editor.
    lamp: Arc<AtomicLevel>,
    /// Reusable control snapshot. Held as plugin state so the per-sample loop
    /// never constructs one on the audio thread.
    controls: SampleControls,
}

impl Default for AmberHeadOr100 {
    fn default() -> Self {
        let params = Arc::new(Or100Params::default());
        let controls = params.settled_controls();
        Self {
            params,
            // The engine carries roughly 40 kB of lookup tables and delay
            // lines; boxing keeps `Plugin` itself small enough to move cheaply.
            engine: Box::new(AmpEngine::new()),
            lamp: Arc::new(AtomicLevel::new(0.0)),
            controls,
        }
    }
}

impl Plugin for AmberHeadOr100 {
    const NAME: &'static str = "AmberHead OR100";
    const VENDOR: &'static str = "AmberHead Audio";
    const URL: &'static str = "https://github.com/JohanDahlberg91/AmberHead_OR100";
    const EMAIL: &'static str = "support@amberhead.audio";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    /// A guitar amplifier is a mono device. Mono-in/mono-out is the primary
    /// layout; mono-in/stereo-out is offered so the plugin can sit on a stereo
    /// track without the host refusing to load it, and stereo-in is accepted
    /// and summed rather than processed as two independent amplifiers — two
    /// amps would double the CPU cost for a signal that is almost always mono.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
    ];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        gui::create(
            self.params.clone(),
            self.lamp.clone(),
            self.params.editor_state.clone(),
        )
    }

    /// Builds every table, filter and buffer the audio path needs.
    ///
    /// This is the *only* place allocation happens: the triode load-line
    /// tables, the oversampling filter designs, the tone stack solutions and
    /// the convolution FFT plans (`CLAUDE.md` §1).
    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.controls = self.params.settled_controls();
        if !self
            .engine
            .prepare(buffer_config.sample_rate, &self.controls)
        {
            nih_error!("Failed to plan the cabinet convolution FFTs");
            return false;
        }

        // Specification section 5: report the bounded FIR/partition latency to
        // the host rather than pretending to be zero-latency.
        context.set_latency_samples(self.engine.latency_samples());
        self.lamp
            .store(self.engine.lamp_brightness(self.controls.power));
        true
    }

    fn reset(&mut self) {
        self.engine.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Flush-to-zero and denormals-are-zero for the whole callback, restored
        // on drop so the host's FPU state is left exactly as it was found.
        let _denormals = DenormalGuard::new();

        // Discrete switch positions are read once per block; only the eight
        // continuous controls are sampled per frame.
        let mut controls = self.params.block_controls();

        for mut channel_samples in buffer.iter_samples() {
            self.params.fill_smoothed(&mut controls);

            // Sum the input channels. A stereo source is almost always a mono
            // guitar that has been duplicated, and running two independent
            // amplifier models would double the CPU cost for no benefit.
            let mut input = 0.0f32;
            let channel_count = channel_samples.len();
            for sample in channel_samples.iter_mut() {
                input += *sample;
            }
            if channel_count > 1 {
                input /= channel_count as f32;
            }

            let output = self.engine.process_sample(input, &controls);

            // The amplifier is mono; every output channel carries the same
            // signal.
            for sample in channel_samples.iter_mut() {
                *sample = output;
            }
        }

        // One atomic store per block, not per sample. The editor samples this
        // at 30 Hz, so per-sample publishing would be pure waste.
        if self.params.editor_state.is_open() {
            self.lamp.store(self.engine.lamp_brightness(controls.power));
        }

        self.controls = controls;
        ProcessStatus::Normal
    }
}

impl ClapPlugin for AmberHeadOr100 {
    const CLAP_ID: &'static str = "audio.amberhead.or100";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Virtual analog recreation of the Orange OR100 Modern Reissue");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Mono,
        ClapFeature::Distortion,
        ClapFeature::Custom("guitar"),
    ];
}

impl Vst3Plugin for AmberHeadOr100 {
    /// Must stay fixed for the lifetime of the product: changing it makes
    /// every saved project fail to find the plugin.
    const VST3_CLASS_ID: [u8; 16] = *b"AmberHeadOR100Va";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Distortion];
}

nih_export_clap!(AmberHeadOr100);
nih_export_vst3!(AmberHeadOr100);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vst3_class_id_is_exactly_sixteen_bytes() {
        assert_eq!(AmberHeadOr100::VST3_CLASS_ID.len(), 16);
    }

    #[test]
    fn plugin_metadata_is_populated() {
        assert!(!AmberHeadOr100::NAME.is_empty());
        assert!(!AmberHeadOr100::VENDOR.is_empty());
        assert!(!AmberHeadOr100::VERSION.is_empty());
        assert!(AmberHeadOr100::CLAP_ID.contains('.'));
    }

    #[test]
    fn every_declared_layout_is_mono_or_stereo() {
        assert!(!AmberHeadOr100::AUDIO_IO_LAYOUTS.is_empty());
        for layout in AmberHeadOr100::AUDIO_IO_LAYOUTS {
            let inputs = layout.main_input_channels.map_or(0, |c| c.get());
            let outputs = layout.main_output_channels.map_or(0, |c| c.get());
            assert!((1..=2).contains(&inputs), "{inputs} input channels");
            assert!((1..=2).contains(&outputs), "{outputs} output channels");
        }
    }

    #[test]
    fn default_construction_yields_a_silent_unprepared_plugin() {
        let plugin = AmberHeadOr100::default();
        assert_eq!(plugin.lamp.load(), 0.0);
        assert_eq!(plugin.controls.dirty_gain, 6.5);
    }
}
