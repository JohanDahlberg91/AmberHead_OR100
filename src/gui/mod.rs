//! Skeuomorphic "Pics Only" amplifier head.
//!
//! Specification section 4. The editor is 920x340 logical pixels and scales
//! from 75 % to 200 %; every control is drawn with vector primitives on the
//! `vizia`/FemtoVG canvas, so there are no raster assets to go soft when the
//! window is scaled.
//!
//! The faceplate reproduces the physical amplifier: an orange vinyl-covered box
//! with moulded black corner protectors, a chrome-framed aperture, a white
//! control panel, and a black-outlined control bar split into a white switch
//! cell and two orange knob cells with pictograms printed above each control.
//! [`chassis`] draws all of it and owns the layout geometry; `theme.css`
//! positions the interactive controls onto the cells it paints.
//!
//! The wordmark, model plate and badge are original. Orange Amplification's
//! logo and coat of arms are that company's trademarks and are not reproduced.
//!
//! This module and its children never touch [`crate::dsp`] types. The only
//! value that crosses the audio/UI boundary is the jewel lamp brightness,
//! carried in a [`crate::shared::AtomicLevel`] (`CLAUDE.md` §3).

pub mod browser;
pub mod chassis;
pub mod glyphs;
pub mod jewel;
pub mod knob;
pub mod switch;
pub mod theme;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use nih_plug::prelude::Editor;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::param_base::ParamWidgetBase;
use nih_plug_vizia::widgets::ResizeHandle;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};

use crate::ir::DirectoryEntry;
use crate::params::Or100Params;
use crate::shared::{AtomicLevel, IrSlot};
use chassis::{Crest, Faceplate};
use glyphs::Glyph;

/// Editor size in logical pixels (specification section 4).
pub const WINDOW_WIDTH: u32 = 920;
/// Editor height in logical pixels (specification section 4).
pub const WINDOW_HEIGHT: u32 = 340;

/// The brand wordmark.
///
/// Spaced character by character because vizia's stylesheet has no
/// `letter-spacing` property, and a logo set solid does not read as one.
const WORDMARK: &str = "A M B E R H E A D";

/// Refresh interval for the jewel lamp. 30 Hz is fast enough to show the
/// rail sagging under an 8 ms attack without spending GPU time redrawing a
/// static faceplate.
const LAMP_REFRESH: Duration = Duration::from_millis(33);

/// Default editor state, used by [`Or100Params`].
pub fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (WINDOW_WIDTH, WINDOW_HEIGHT))
}

/// Editor model.
///
/// The impulse response fields are ordinary editor state: `CLAUDE.md` section 3
/// keeps the GUI away from DSP types, so the browser talks to the audio thread
/// only through [`IrSlot`], which carries plain `f32` taps and knows nothing
/// about the convolver on the other side.
#[derive(Lens)]
pub struct Data {
    pub(crate) params: Arc<Or100Params>,
    /// Written by the audio thread, sampled on the timer tick below.
    lamp: Arc<AtomicLevel>,
    /// Copy of `lamp` held as plain state, so changing it marks the lens dirty
    /// and schedules a redraw. Reading the atomic directly inside `draw` would
    /// never trigger one.
    lamp_brightness: f32,
    /// Host sample rate, published by `Plugin::initialize`. The browser needs
    /// it in order to resample a loaded response.
    pub(crate) sample_rate: Arc<AtomicLevel>,
    /// Where a loaded response is handed to the audio thread.
    pub(crate) ir_slot: Arc<IrSlot>,

    /// Whether the browser overlay is showing.
    pub(crate) browser_open: bool,
    /// Directory the browser is listing.
    pub(crate) ir_location: PathBuf,
    /// That directory, rendered for the path strip.
    pub(crate) ir_directory: String,
    /// Rows of the current listing.
    pub(crate) ir_entries: Vec<DirectoryEntry>,
    /// Caption on the launcher button.
    pub(crate) ir_cabinet: String,
    /// Last thing the browser has to say, shown along the bottom of the panel.
    pub(crate) ir_status: String,
}

/// Editor-internal events.
pub enum UiEvent {
    /// Sample the shared lamp level.
    RefreshLamp,
    /// Show or hide the impulse response browser.
    ToggleBrowser,
    /// Move to the parent directory.
    BrowseUp,
    /// Act on the row at this index of the current listing.
    SelectEntry(usize),
    /// Go back to the synthesised 4x12.
    UseDefaultCabinet,
}

impl Model for Data {
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|ui_event, _meta| match ui_event {
            UiEvent::RefreshLamp => {
                self.lamp_brightness = self.lamp.load().clamp(0.0, 1.0);
            }
            UiEvent::ToggleBrowser => self.toggle_browser(),
            UiEvent::BrowseUp => self.browse_up(),
            UiEvent::SelectEntry(index) => self.select_entry(*index),
            UiEvent::UseDefaultCabinet => self.use_default_cabinet(),
        });
    }
}

/// Builds the editor.
pub fn create(
    params: Arc<Or100Params>,
    lamp: Arc<AtomicLevel>,
    sample_rate: Arc<AtomicLevel>,
    ir_slot: Arc<IrSlot>,
    editor_state: Arc<ViziaState>,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(
        editor_state,
        ViziaTheming::Custom,
        move |cx, _gui_context| {
            assets::register_noto_sans_light(cx);
            assets::register_noto_sans_thin(cx);
            // The wordmark is set in bold; without this it silently falls back
            // to the light face and the header reads as body text.
            assets::register_noto_sans_bold(cx);
            if let Err(error) = cx.add_stylesheet(include_style!("src/gui/theme.css")) {
                nih_plug::nih_error!("Failed to load the faceplate stylesheet: {error:?}");
            }

            // The persisted path is the source of truth for which cabinet is
            // loaded; the engine has already applied it in `initialize`, so the
            // editor only has to describe it.
            let ir_path = params
                .ir_path
                .read()
                .map(|path| path.clone())
                .unwrap_or_default();
            let location = crate::ir::starting_directory(&ir_path);

            Data {
                params: params.clone(),
                lamp: lamp.clone(),
                lamp_brightness: lamp.load(),
                sample_rate: sample_rate.clone(),
                ir_slot: ir_slot.clone(),
                browser_open: false,
                ir_directory: browser::display_directory(&location),
                ir_location: location,
                ir_entries: Vec::new(),
                ir_cabinet: browser::cabinet_label(&ir_path),
                ir_status: String::new(),
            }
            .build(cx);

            let timer = cx.add_timer(LAMP_REFRESH, None, |cx, action| {
                if let TimerAction::Tick(_) = action {
                    cx.emit(UiEvent::RefreshLamp);
                }
            });
            cx.start_timer(timer);

            ZStack::new(cx, |cx| {
                Faceplate.build(cx, |_| {}).class("faceplate");
                build_controls(cx);
                browser::browser_overlay(cx);
            });

            ResizeHandle::new(cx);
        },
    )
}

/// Lays the header and the three control-bar cells onto the painted chassis.
///
/// Both containers are absolutely positioned onto the geometry in [`chassis`];
/// the stylesheet carries the same numbers, and `chassis::tests` checks that
/// the cells tile the bar exactly so a knob can never end up half on the white
/// panel and half on an orange cell.
fn build_controls(cx: &mut Context) {
    // --- Header: wordmark, cabinet plate, model plate, badge ---------------
    HStack::new(cx, |cx| {
        Label::new(cx, WORDMARK).class("wordmark");
        Element::new(cx).class("header-spacer");
        browser::cabinet_button(cx);
        Label::new(cx, "OR100").class("model-mark");
        Crest.build(cx, |_| {}).class("crest");
    })
    .class("header");

    // --- Control bar --------------------------------------------------------
    HStack::new(cx, |cx| {
        // Left cell, on white: the pilot jewel and the six switches, in the
        // two rows the real amplifier's switch bank uses.
        HStack::new(cx, |cx| {
            jewel::jewel_lamp(cx, Data::lamp_brightness);
            VStack::new(cx, |cx| {
                HStack::new(cx, |cx| {
                    bat_cell(cx, |p| &p.channel, "CHAN");
                    bat_cell(cx, |p| &p.gain_boost, "BOOST");
                    bat_cell(cx, |p| &p.global_boost, "+3dB");
                })
                .class("switch-row");
                HStack::new(cx, |cx| {
                    bat_cell(cx, |p| &p.cab_enabled, "CAB");
                    lever_cell(cx, |p| &p.power_switch, "PWR");
                    lever_cell(cx, |p| &p.output_power, "WATT");
                })
                .class("switch-row");
            })
            .class("switch-grid");
        })
        // `Handle::class` inserts the whole string as a single class name, so
        // the two classes have to be applied separately; "cell cell-switches"
        // would register one class of that name and match no rule at all.
        .class("cell")
        .class("cell-switches");

        // Middle cell, on orange: the clean channel.
        HStack::new(cx, |cx| {
            knob_cell(cx, |p| &p.clean_volume, Glyph::Speaker);
            knob_cell(cx, |p| &p.clean_bass, Glyph::BassClef);
            knob_cell(cx, |p| &p.clean_treble, Glyph::TrebleClef);
        })
        .class("cell")
        .class("cell-clean");

        // Right cell, on orange: the dirty channel.
        HStack::new(cx, |cx| {
            knob_cell(cx, |p| &p.dirty_gain, Glyph::Burst);
            knob_cell(cx, |p| &p.dirty_bass, Glyph::BassClef);
            knob_cell(cx, |p| &p.dirty_middle, Glyph::SoundWave);
            knob_cell(cx, |p| &p.dirty_treble, Glyph::TrebleClef);
            knob_cell(cx, |p| &p.dirty_volume, Glyph::Speaker);
        })
        .class("cell")
        .class("cell-dirty");
    })
    .class("control-bar");
}

/// One knob plus its numeric readout.
fn knob_cell<P, FMap>(cx: &mut Context, params_to_param: FMap, glyph: Glyph)
where
    P: nih_plug::prelude::Param + 'static,
    FMap: Fn(&Arc<Or100Params>) -> &P + Copy + 'static,
{
    VStack::new(cx, |cx| {
        knob::knob(cx, Data::params, params_to_param, glyph);
        Label::new(
            cx,
            ParamWidgetBase::make_lens(Data::params, params_to_param, |param| {
                // `Param::Plain` is an associated type, so the parameter's own
                // formatter is used rather than `Display`. It renders to two
                // decimals, matching the 0.01 step size the schema declares.
                param.normalized_value_to_string(param.modulated_normalized_value(), false)
            }),
        )
        .class("value-readout");
    })
    .class("knob-cell");
}

/// One bat switch plus its caption.
fn bat_cell<P, FMap>(cx: &mut Context, params_to_param: FMap, caption: &'static str)
where
    P: nih_plug::prelude::Param + 'static,
    FMap: Fn(&Arc<Or100Params>) -> &P + Copy + 'static,
{
    VStack::new(cx, |cx| {
        switch::bat_switch(cx, Data::params, params_to_param);
        Label::new(cx, caption).class("switch-caption");
    })
    .class("switch-cell");
}

/// One multi-position lever switch plus its caption and current setting.
fn lever_cell<P, FMap>(cx: &mut Context, params_to_param: FMap, caption: &'static str)
where
    P: nih_plug::prelude::Param + 'static,
    FMap: Fn(&Arc<Or100Params>) -> &P + Copy + 'static,
{
    VStack::new(cx, |cx| {
        switch::lever_switch(cx, Data::params, params_to_param);
        Label::new(cx, caption).class("switch-caption");
        Label::new(
            cx,
            ParamWidgetBase::make_lens(Data::params, params_to_param, |param| {
                param.normalized_value_to_string(param.modulated_normalized_value(), false)
            }),
        )
        .class("switch-value");
    })
    .class("switch-cell");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an editor model without a live `vizia` context, so the parts of
    /// it that are plain state can be unit-tested.
    fn model_with(params: Arc<Or100Params>, lamp: Arc<AtomicLevel>) -> Data {
        let location = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Data {
            params,
            lamp_brightness: lamp.load(),
            lamp,
            sample_rate: Arc::new(AtomicLevel::new(48_000.0)),
            ir_slot: Arc::new(IrSlot::new(crate::dsp::cabinet::IR_LENGTH)),
            browser_open: false,
            ir_directory: browser::display_directory(&location),
            ir_location: location,
            ir_entries: Vec::new(),
            ir_cabinet: browser::cabinet_label(""),
            ir_status: String::new(),
        }
    }

    /// Writes a minimal 32-bit float WAV and returns its path.
    fn write_test_wav(name: &str, rate: u32, samples: &[f32]) -> PathBuf {
        let data: Vec<u8> = samples
            .iter()
            .flat_map(|sample| sample.to_bits().to_le_bytes())
            .collect();

        let mut fmt: Vec<u8> = Vec::new();
        fmt.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        fmt.extend_from_slice(&1u16.to_le_bytes()); // mono
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * 4).to_le_bytes());
        fmt.extend_from_slice(&4u16.to_le_bytes());
        fmt.extend_from_slice(&32u16.to_le_bytes());

        let mut body: Vec<u8> = b"WAVE".to_vec();
        body.extend_from_slice(b"fmt ");
        body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        body.extend_from_slice(&fmt);
        body.extend_from_slice(b"data");
        body.extend_from_slice(&(data.len() as u32).to_le_bytes());
        body.extend_from_slice(&data);

        let mut file: Vec<u8> = b"RIFF".to_vec();
        file.extend_from_slice(&(body.len() as u32).to_le_bytes());
        file.extend_from_slice(&body);

        let mut path = std::env::temp_dir();
        path.push(name);
        std::fs::write(&path, &file).expect("could not write the test WAV");
        path
    }

    #[test]
    fn browsing_lists_a_directory_and_walks_back_out_of_it() {
        let params = Arc::new(Or100Params::default());
        let mut model = model_with(params, Arc::new(AtomicLevel::new(0.0)));

        let here = std::env::current_dir().expect("no working directory");
        model.browse_to(here.clone());
        assert_eq!(model.ir_location, here);
        assert!(!model.ir_entries.is_empty(), "the repository listed empty");
        assert!(model.ir_directory.contains("AmberHead"));

        model.browse_up();
        assert_eq!(Some(model.ir_location.as_path()), here.parent());

        // Walking to the very top must stop rather than loop.
        for _ in 0..32 {
            model.browse_up();
        }
        assert_eq!(model.ir_status, "already at the top");

        // An unreadable directory reports why and leaves the listing alone.
        let entries = model.ir_entries.clone();
        model.browse_to(PathBuf::from("/no/such/directory/anywhere"));
        assert!(model.ir_status.starts_with("cannot open folder"));
        assert_eq!(model.ir_entries, entries);
    }

    #[test]
    fn selecting_a_folder_navigates_and_selecting_a_file_loads() {
        let params = Arc::new(Or100Params::default());
        let mut model = model_with(params.clone(), Arc::new(AtomicLevel::new(0.0)));

        let path = write_test_wav(
            "amberhead_or100_gui_load.wav",
            48_000,
            &(0..256).map(|n| 0.9f32.powi(n) * 0.5).collect::<Vec<f32>>(),
        );
        let directory = path.parent().expect("no parent").to_path_buf();
        model.browse_to(directory);

        let index = model
            .ir_entries
            .iter()
            .position(|entry| entry.path == path)
            .expect("the test WAV was not listed");
        assert!(!model.ir_entries.get(index).expect("listed").is_directory);

        let mut seen = model.ir_slot.current_generation();
        let mut taps = vec![0.0f32; crate::dsp::cabinet::IR_LENGTH];
        model.select_entry(index);

        assert_eq!(model.ir_slot.collect(&mut seen, &mut taps), Some(256));
        assert_eq!(
            model.ir_cabinet,
            browser::cabinet_label(&path.to_string_lossy())
        );
        assert_ne!(model.ir_cabinet, browser::DEFAULT_CABINET_LABEL);
        assert!(model.ir_status.contains("256 taps"), "{}", model.ir_status);
        assert!(!model.browser_open, "loading left the browser open");
        assert_eq!(
            params.ir_path.read().map(|p| p.clone()).unwrap_or_default(),
            path.to_string_lossy()
        );

        // Selecting a directory navigates instead of loading.
        let up = model.ir_entries.iter().position(|entry| entry.is_directory);
        if let Some(up) = up {
            let target = model.ir_entries.get(up).expect("listed").path.clone();
            model.select_entry(up);
            assert_eq!(model.ir_location, target);
        }

        // An out-of-range index is ignored rather than panicking.
        model.select_entry(usize::MAX);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn loading_is_refused_before_the_host_reports_a_sample_rate() {
        let params = Arc::new(Or100Params::default());
        let mut model = model_with(params.clone(), Arc::new(AtomicLevel::new(0.0)));
        // `model_with` seeds 48 kHz; clear it to stand in for an editor opened
        // before `initialize` has run.
        model.sample_rate.store(0.0);

        let path = write_test_wav("amberhead_or100_gui_early.wav", 48_000, &[1.0, 0.5]);
        model.load_cabinet(&path);
        assert_eq!(model.ir_status, "the host has not started audio yet");
        assert_eq!(
            params.ir_path.read().map(|p| p.clone()).unwrap_or_default(),
            "",
            "a refused load still wrote the persisted path"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_rejected_file_reports_the_reason_and_changes_nothing() {
        let params = Arc::new(Or100Params::default());
        let mut model = model_with(params.clone(), Arc::new(AtomicLevel::new(0.0)));

        let mut path = std::env::temp_dir();
        path.push("amberhead_or100_gui_not_a_wav.wav");
        std::fs::write(&path, b"this is plainly not a wave file").expect("write failed");

        let before = model.ir_cabinet.clone();
        model.load_cabinet(&path);
        assert!(
            model.ir_status.contains("not a RIFF/WAVE file"),
            "{}",
            model.ir_status
        );
        assert_eq!(
            model.ir_cabinet, before,
            "a failed load renamed the cabinet"
        );
        assert_eq!(
            params.ir_path.read().map(|p| p.clone()).unwrap_or_default(),
            ""
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reverting_to_the_built_in_cabinet_publishes_an_empty_payload() {
        let params = Arc::new(Or100Params::default());
        if let Ok(mut stored) = params.ir_path.write() {
            *stored = "C:/cabs/whatever.wav".into();
        }
        let mut model = model_with(params.clone(), Arc::new(AtomicLevel::new(0.0)));

        let mut seen = model.ir_slot.current_generation();
        let mut taps = vec![0.0f32; 16];
        model.use_default_cabinet();

        assert_eq!(model.ir_slot.collect(&mut seen, &mut taps), Some(0));
        assert_eq!(model.ir_cabinet, browser::DEFAULT_CABINET_LABEL);
        assert_eq!(
            params.ir_path.read().map(|p| p.clone()).unwrap_or_default(),
            "",
            "the persisted path was not cleared"
        );
    }

    #[test]
    fn toggling_the_browser_refreshes_the_listing_only_when_opening() {
        let params = Arc::new(Or100Params::default());
        let mut model = model_with(params, Arc::new(AtomicLevel::new(0.0)));
        assert!(!model.browser_open);
        assert!(model.ir_entries.is_empty());

        model.toggle_browser();
        assert!(model.browser_open);
        assert!(
            !model.ir_entries.is_empty(),
            "opening did not list anything"
        );

        model.toggle_browser();
        assert!(!model.browser_open);
    }

    #[test]
    fn default_state_reports_the_specified_size() {
        let state = default_state();
        assert_eq!(state.inner_logical_size(), (WINDOW_WIDTH, WINDOW_HEIGHT));
        assert_eq!((WINDOW_WIDTH, WINDOW_HEIGHT), (920, 340));
    }

    /// The stylesheet, read at compile time so it can be checked against the
    /// geometry `chassis` paints.
    const STYLESHEET: &str = include_str!("theme.css");

    /// Reads a pixel-valued declaration out of one CSS rule.
    ///
    /// A deliberately small parser: it only has to understand the handful of
    /// `property: Npx;` lines this file writes, and anything it fails to find
    /// fails the test rather than passing silently.
    fn css_pixels(selector: &str, property: &str) -> f32 {
        let header = format!("{selector} {{");
        let start = STYLESHEET
            .find(&header)
            .unwrap_or_else(|| panic!("no `{selector}` rule in theme.css"))
            + header.len();
        let rest = STYLESHEET.get(start..).unwrap_or_default();
        let end = rest
            .find('}')
            .unwrap_or_else(|| panic!("`{selector}` rule is unterminated"));
        let block = rest.get(..end).unwrap_or_default();

        for line in block.lines() {
            let Some(value) = line.trim().strip_prefix(&format!("{property}:")) else {
                continue;
            };
            let value = value
                .trim()
                .trim_end_matches(';')
                .trim()
                .trim_end_matches("px");
            return value
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("`{selector}` has an unparseable `{property}`"));
        }
        panic!("`{selector}` declares no `{property}`");
    }

    #[test]
    fn the_stylesheet_agrees_with_the_painted_chassis() {
        // The single failure this guards against: a container positioned by the
        // stylesheet drifting off the surface `chassis` paints under it, which
        // puts a knob half on the white panel and half on an orange cell.
        for (selector, plate) in [
            ("\n.header", chassis::HEADER),
            ("\n.control-bar", chassis::BAR),
        ] {
            assert_eq!(css_pixels(selector, "left"), plate.x, "{selector} left");
            assert_eq!(css_pixels(selector, "top"), plate.y, "{selector} top");
            assert_eq!(css_pixels(selector, "width"), plate.w, "{selector} width");
            assert_eq!(css_pixels(selector, "height"), plate.h, "{selector} height");
        }

        for (selector, plate) in [
            (".cell-switches", chassis::CELL_SWITCHES),
            (".cell-clean", chassis::CELL_CLEAN),
            (".cell-dirty", chassis::CELL_DIRTY),
        ] {
            assert_eq!(css_pixels(selector, "width"), plate.w, "{selector} width");
        }
    }

    #[test]
    fn the_wordmark_is_spaced_out_but_still_spells_the_brand() {
        assert_eq!(WORDMARK.replace(' ', ""), "AMBERHEAD");
        // Every character is separated, which is what stands in for the
        // `letter-spacing` property vizia does not have.
        assert!(
            WORDMARK.chars().skip(1).step_by(2).all(|c| c == ' '),
            "the wordmark is not evenly spaced: {WORDMARK}"
        );
    }

    #[test]
    fn lamp_refresh_is_fast_enough_to_show_sag() {
        // The sag envelope attacks in 8 ms and releases in 120 ms; the refresh
        // must comfortably resolve the release.
        assert!(LAMP_REFRESH.as_millis() < 120 / 3);
    }

    #[test]
    fn model_clamps_the_shared_lamp_level() {
        let lamp = Arc::new(AtomicLevel::new(5.0));
        let mut data = model_with(Arc::new(Or100Params::default()), lamp.clone());
        // Emulate the timer tick without a live event context.
        data.lamp_brightness = data.lamp.load().clamp(0.0, 1.0);
        assert_eq!(data.lamp_brightness, 1.0);

        lamp.store(-2.0);
        data.lamp_brightness = data.lamp.load().clamp(0.0, 1.0);
        assert_eq!(data.lamp_brightness, 0.0);
    }
}
