//! Standalone JACK/CoreAudio/WASAPI host wrapper.
//!
//! Specification section 1 lists Standalone alongside VST3 and CLAP as a
//! delivery target. `nih_export_standalone` provides the backend selection,
//! device configuration and command-line parsing.

use nih_plug::prelude::nih_export_standalone;

use amberhead_or100::AmberHeadOr100;

fn main() {
    nih_export_standalone::<AmberHeadOr100>();
}
