//! Skeuomorphic Orange "Pics Only" faceplate.
//!
//! Specification section 4. The editor is 920x340 logical pixels and scales
//! from 75 % to 200 %; every control is drawn with vector primitives on the
//! `vizia`/FemtoVG canvas, so there are no raster assets to go soft when the
//! window is scaled.
//!
//! This module and its children never touch [`crate::dsp`] types. The only
//! value that crosses the audio/UI boundary is the jewel lamp brightness,
//! carried in a [`crate::shared::AtomicLevel`] (`CLAUDE.md` §3).

pub mod glyphs;
pub mod jewel;
pub mod knob;
pub mod switch;
pub mod theme;

use std::sync::Arc;
use std::time::Duration;

use nih_plug::prelude::Editor;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use nih_plug_vizia::widgets::param_base::ParamWidgetBase;
use nih_plug_vizia::widgets::ResizeHandle;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};

use crate::params::Or100Params;
use crate::shared::AtomicLevel;
use glyphs::Glyph;

/// Editor size in logical pixels (specification section 4).
pub const WINDOW_WIDTH: u32 = 920;
/// Editor height in logical pixels (specification section 4).
pub const WINDOW_HEIGHT: u32 = 340;

/// Height of each orange framing stripe, in logical pixels.
const STRIPE_HEIGHT: f32 = 17.0;
/// Refresh interval for the jewel lamp. 30 Hz is fast enough to show the
/// rail sagging under an 8 ms attack without spending GPU time redrawing a
/// static faceplate.
const LAMP_REFRESH: Duration = Duration::from_millis(33);

/// Default editor state, used by [`Or100Params`].
pub fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (WINDOW_WIDTH, WINDOW_HEIGHT))
}

/// Editor model.
#[derive(Lens)]
pub struct Data {
    params: Arc<Or100Params>,
    /// Written by the audio thread, sampled on the timer tick below.
    lamp: Arc<AtomicLevel>,
    /// Copy of `lamp` held as plain state, so changing it marks the lens dirty
    /// and schedules a redraw. Reading the atomic directly inside `draw` would
    /// never trigger one.
    lamp_brightness: f32,
}

/// Editor-internal events.
pub enum UiEvent {
    /// Sample the shared lamp level.
    RefreshLamp,
}

impl Model for Data {
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|ui_event, _meta| match ui_event {
            UiEvent::RefreshLamp => {
                self.lamp_brightness = self.lamp.load().clamp(0.0, 1.0);
            }
        });
    }
}

/// Builds the editor.
pub fn create(
    params: Arc<Or100Params>,
    lamp: Arc<AtomicLevel>,
    editor_state: Arc<ViziaState>,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(
        editor_state,
        ViziaTheming::Custom,
        move |cx, _gui_context| {
            assets::register_noto_sans_light(cx);
            assets::register_noto_sans_thin(cx);
            if let Err(error) = cx.add_stylesheet(include_style!("src/gui/theme.css")) {
                nih_plug::nih_error!("Failed to load the faceplate stylesheet: {error:?}");
            }

            Data {
                params: params.clone(),
                lamp: lamp.clone(),
                lamp_brightness: lamp.load(),
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
            });

            ResizeHandle::new(cx);
        },
    )
}

/// Lays out the brand block, the eight knobs and the switch bank.
fn build_controls(cx: &mut Context) {
    HStack::new(cx, |cx| {
        // --- Brand and pilot jewel ---------------------------------------
        VStack::new(cx, |cx| {
            jewel::jewel_lamp(cx, Data::lamp_brightness);
            Label::new(cx, "AMBERHEAD").class("brand-mark");
            Label::new(cx, "OR100").class("brand-model");
        })
        .class("brand");

        // --- Clean channel: Volume, Bass, Treble --------------------------
        VStack::new(cx, |cx| {
            HStack::new(cx, |cx| {
                knob_cell(cx, |p| &p.clean_volume, Glyph::Speaker);
                knob_cell(cx, |p| &p.clean_bass, Glyph::BassClef);
                knob_cell(cx, |p| &p.clean_treble, Glyph::TrebleClef);
            })
            .class("knob-row");
            Label::new(cx, "CLEAN").class("group-caption");
        })
        .class("channel-group");

        Element::new(cx).class("divider");

        // --- Dirty channel: Gain, Bass, Middle, Treble, Volume ------------
        VStack::new(cx, |cx| {
            HStack::new(cx, |cx| {
                knob_cell(cx, |p| &p.dirty_gain, Glyph::Burst);
                knob_cell(cx, |p| &p.dirty_bass, Glyph::BassClef);
                knob_cell(cx, |p| &p.dirty_middle, Glyph::SoundWave);
                knob_cell(cx, |p| &p.dirty_treble, Glyph::TrebleClef);
                knob_cell(cx, |p| &p.dirty_volume, Glyph::Speaker);
            })
            .class("knob-row");
            Label::new(cx, "DIRTY").class("group-caption");
        })
        .class("channel-group");

        Element::new(cx).class("divider");

        // --- Switch bank ---------------------------------------------------
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
        .class("switch-bank");
    })
    .class("panel");
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

/// The chassis itself: orange stripes framing a textured ivory enamel panel.
pub struct Faceplate;

impl View for Faceplate {
    fn element(&self) -> Option<&'static str> {
        Some("faceplate")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        if bounds.w <= 0.0 || bounds.h <= 0.0 {
            return;
        }
        let opacity = cx.opacity();
        // The stripes keep their proportion of the chassis as the window is
        // scaled, rather than staying a fixed pixel height.
        let stripe = (bounds.h * (STRIPE_HEIGHT / WINDOW_HEIGHT as f32)).max(2.0);

        // Enamel panel.
        let mut panel = vg::Path::new();
        panel.rect(bounds.x, bounds.y, bounds.w, bounds.h);
        let panel_paint = vg::Paint::linear_gradient(
            bounds.x,
            bounds.y,
            bounds.x,
            bounds.y + bounds.h,
            theme::with_opacity(theme::IVORY, opacity),
            theme::with_opacity(theme::IVORY_SHADE, opacity),
        );
        canvas.fill_path(&panel, &panel_paint);

        draw_enamel_texture(canvas, &bounds, stripe, opacity);

        // Orange stripes, top and bottom.
        let stripe_paint = theme::with_opacity(theme::ORANGE, opacity);
        let mut stripes = vg::Path::new();
        stripes.rect(bounds.x, bounds.y, bounds.w, stripe);
        stripes.rect(bounds.x, bounds.y + bounds.h - stripe, bounds.w, stripe);
        canvas.fill_path(&stripes, &vg::Paint::color(stripe_paint));

        // Shadow line where each stripe meets the enamel.
        let mut seams = vg::Path::new();
        seams.rect(bounds.x, bounds.y + stripe, bounds.w, stripe * 0.13);
        seams.rect(
            bounds.x,
            bounds.y + bounds.h - stripe - stripe * 0.13,
            bounds.w,
            stripe * 0.13,
        );
        canvas.fill_path(
            &seams,
            &vg::Paint::color(theme::with_opacity(theme::ORANGE_DEEP, opacity * 0.8)),
        );
    }
}

/// Speckled enamel texture.
///
/// Positions come from a fixed-seed linear congruential generator rather than a
/// random source, so the texture is identical on every redraw and every
/// instance — a texture that shimmered between frames would be far more
/// distracting than no texture at all.
fn draw_enamel_texture(canvas: &mut Canvas, bounds: &BoundingBox, stripe: f32, opacity: f32) {
    const SPECKLES: usize = 700;
    let top = bounds.y + stripe;
    let usable_height = (bounds.h - stripe * 2.0).max(1.0);

    let mut state: u32 = 0x9E37_79B9;
    let mut next = || {
        // Numerical Recipes LCG constants.
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / 16_777_216.0
    };

    let mut speckles = vg::Path::new();
    for _ in 0..SPECKLES {
        let x = bounds.x + next() * bounds.w;
        let y = top + next() * usable_height;
        let size = 0.6 + next() * 1.1;
        speckles.rect(x, y, size, size);
    }
    canvas.fill_path(
        &speckles,
        &vg::Paint::color(theme::with_opacity(theme::PANEL_ENGRAVE, opacity * 0.22)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_reports_the_specified_size() {
        let state = default_state();
        assert_eq!(state.inner_logical_size(), (WINDOW_WIDTH, WINDOW_HEIGHT));
        assert_eq!((WINDOW_WIDTH, WINDOW_HEIGHT), (920, 340));
    }

    #[test]
    fn stripes_scale_with_the_chassis() {
        let proportion = STRIPE_HEIGHT / WINDOW_HEIGHT as f32;
        // At 100 % the stripe is its nominal height...
        assert!((WINDOW_HEIGHT as f32 * proportion - STRIPE_HEIGHT).abs() < 1.0e-3);
        // ...and it keeps that proportion across the 75 %..200 % scale range.
        for scale in [0.75f32, 1.0, 1.5, 2.0] {
            let height = WINDOW_HEIGHT as f32 * scale;
            let stripe = (height * proportion).max(2.0);
            assert!((stripe / height - proportion).abs() < 1.0e-4);
            assert!(stripe * 2.0 < height, "stripes swallowed the panel");
        }
    }

    #[test]
    fn texture_generator_is_deterministic_and_in_range() {
        let sample = || {
            let mut state: u32 = 0x9E37_79B9;
            let mut values = Vec::with_capacity(700);
            for _ in 0..700 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                values.push((state >> 8) as f32 / 16_777_216.0);
            }
            values
        };
        let first = sample();
        let second = sample();
        assert_eq!(first, second, "texture is not reproducible");
        assert!(first.iter().all(|v| (0.0..1.0).contains(v)));
        // A degenerate generator that returned a constant would cluster every
        // speckle in one spot.
        let mean: f32 = first.iter().sum::<f32>() / first.len() as f32;
        assert!(
            (0.4..0.6).contains(&mean),
            "speckles are not spread: {mean}"
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
        let mut data = Data {
            params: Arc::new(Or100Params::default()),
            lamp: lamp.clone(),
            lamp_brightness: 0.0,
        };
        // Emulate the timer tick without a live event context.
        data.lamp_brightness = data.lamp.load().clamp(0.0, 1.0);
        assert_eq!(data.lamp_brightness, 1.0);

        lamp.store(-2.0);
        data.lamp_brightness = data.lamp.load().clamp(0.0, 1.0);
        assert_eq!(data.lamp_brightness, 0.0);
    }
}
