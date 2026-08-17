//! Metal toggle hardware.
//!
//! Specification section 4: "3-way vertical metal switches for Power
//! (Full/Standby/Half) and 2-way bat switches for Boosts and Channels."
//!
//! Two widgets live here:
//!
//! * [`bat_switch`] — a two-position chrome bat toggle bound to a `BoolParam`,
//!   used for the channel selector, both boosts and the cabinet bypass.
//! * [`lever_switch`] — an N-position vertical lever bound to an `EnumParam`.
//!   The power/standby switch has three positions; the wattage selector has
//!   four, because the switching matrix in specification section 2.C defines
//!   four modes and a three-position lever cannot address them.
//!
//! Both emit parameter changes through [`ParamWidgetBase`], which forwards them
//! as `RawParamEvent`s for the `nih_plug_vizia` wrapper to apply. No audio
//! thread lock is taken anywhere in this module (`CLAUDE.md` §3).

use nih_plug::prelude::Param;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use nih_plug_vizia::widgets::param_base::ParamWidgetBase;

use super::theme;

/// Height of the topmost detent, as a fraction of the plate.
const LEVER_TRAVEL_TOP: f32 = 0.16;
/// Distance the lever tip sweeps between the top and bottom detents, as a
/// fraction of the plate. Together with [`LEVER_TRAVEL_TOP`] this keeps the tip
/// inside the plate at both extremes.
const LEVER_TRAVEL_SPAN: f32 = 0.68;

/// A two-position chrome bat toggle.
pub struct BatSwitch<L>
where
    L: Lens<Target = bool>,
{
    param: ParamWidgetBase,
    engaged: L,
}

/// Creates a bat switch bound to a `BoolParam`.
pub fn bat_switch<L, Params, P, FMap>(
    cx: &mut Context,
    params: L,
    params_to_param: FMap,
) -> Handle<'_, impl View>
where
    L: Lens<Target = Params> + Clone,
    Params: 'static,
    P: Param + 'static,
    FMap: Fn(&Params) -> &P + Copy + 'static,
{
    let param = ParamWidgetBase::new(cx, params, params_to_param);
    // A `BoolParam` normalizes to 0.0 or 1.0; anything at or above the midpoint
    // counts as engaged.
    let engaged = ParamWidgetBase::make_lens(params, params_to_param, |param| {
        param.modulated_normalized_value() >= 0.5
    });

    BatSwitch { param, engaged }
        .build(cx, |_| {})
        .class("bat-switch")
}

impl<L> View for BatSwitch<L>
where
    L: Lens<Target = bool>,
{
    fn element(&self) -> Option<&'static str> {
        Some("bat-switch")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|window_event, meta| match window_event {
            WindowEvent::MouseDown(MouseButton::Left)
            | WindowEvent::MouseDoubleClick(MouseButton::Left)
            | WindowEvent::MouseTripleClick(MouseButton::Left) => {
                let currently_on = self.param.modulated_normalized_value() >= 0.5;
                self.param.begin_set_parameter(cx);
                self.param
                    .set_normalized_value(cx, if currently_on { 0.0 } else { 1.0 });
                self.param.end_set_parameter(cx);
                meta.consume();
            }
            _ => {}
        });
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        if bounds.w <= 0.0 || bounds.h <= 0.0 {
            return;
        }
        let opacity = cx.opacity();
        let engaged = self.engaged.get(cx);

        draw_base_plate(canvas, bounds.x, bounds.y, bounds.w, bounds.h, opacity);

        // Two detents: down (off) at the bottom, up (on) at the top.
        let travel = if engaged { 0.0 } else { 1.0 };
        draw_lever(
            canvas,
            bounds.x + bounds.w * 0.5,
            bounds.y,
            bounds.h,
            bounds.w,
            travel,
            opacity,
        );
    }
}

/// An N-position vertical lever bound to an `EnumParam`.
pub struct LeverSwitch<L>
where
    L: Lens<Target = f32>,
{
    param: ParamWidgetBase,
    /// Normalized parameter value, `0.0..=1.0`.
    normalized: L,
    /// Number of selectable positions.
    positions: usize,
}

/// Creates a lever switch bound to an `EnumParam`.
///
/// The number of detents is taken from the parameter's own step count, so
/// adding a variant to the enum automatically adds a detent.
pub fn lever_switch<L, Params, P, FMap>(
    cx: &mut Context,
    params: L,
    params_to_param: FMap,
) -> Handle<'_, impl View>
where
    L: Lens<Target = Params> + Clone,
    Params: 'static,
    P: Param + 'static,
    FMap: Fn(&Params) -> &P + Copy + 'static,
{
    let param = ParamWidgetBase::new(cx, params, params_to_param);
    // `step_count()` is one less than the number of discrete values. The
    // fallback of two keeps a continuous parameter usable as a plain toggle
    // rather than dividing by zero.
    let positions = param.step_count().map_or(2, |steps| steps + 1).max(2);
    let normalized = ParamWidgetBase::make_lens(params, params_to_param, |param| {
        param.modulated_normalized_value()
    });

    LeverSwitch {
        param,
        normalized,
        positions,
    }
    .build(cx, |_| {})
    .class("lever-switch")
}

impl<L> LeverSwitch<L>
where
    L: Lens<Target = f32>,
{
    /// Selects the detent nearest to a cursor position inside the widget.
    ///
    /// The topmost detent is the *last* enum variant so the lever reads like a
    /// real panel switch, where pushing the bat up selects the higher setting.
    fn select_from_cursor(&self, cx: &mut EventContext, cursor_y: f32) {
        let bounds = cx.bounds();
        if bounds.h <= 0.0 {
            return;
        }
        let travel = ((cursor_y - bounds.y) / bounds.h).clamp(0.0, 1.0);
        let last = (self.positions - 1) as f32;
        let index = ((1.0 - travel) * last).round().clamp(0.0, last);

        self.param.begin_set_parameter(cx);
        self.param.set_normalized_value(cx, index / last);
        self.param.end_set_parameter(cx);
    }
}

impl<L> View for LeverSwitch<L>
where
    L: Lens<Target = f32>,
{
    fn element(&self) -> Option<&'static str> {
        Some("lever-switch")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|window_event, meta| match window_event {
            WindowEvent::MouseDown(MouseButton::Left)
            | WindowEvent::MouseDoubleClick(MouseButton::Left)
            | WindowEvent::MouseTripleClick(MouseButton::Left) => {
                let cursor_y = cx.mouse().cursory;
                self.select_from_cursor(cx, cursor_y);
                meta.consume();
            }
            WindowEvent::MouseScroll(_horizontal, vertical) => {
                let current = self.param.unmodulated_normalized_value();
                let next = if *vertical > 0.0 {
                    self.param.next_normalized_step(current, false)
                } else if *vertical < 0.0 {
                    self.param.previous_normalized_step(current, false)
                } else {
                    current
                };
                self.param.begin_set_parameter(cx);
                self.param.set_normalized_value(cx, next);
                self.param.end_set_parameter(cx);
                meta.consume();
            }
            _ => {}
        });
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        if bounds.w <= 0.0 || bounds.h <= 0.0 {
            return;
        }
        let opacity = cx.opacity();

        draw_base_plate(canvas, bounds.x, bounds.y, bounds.w, bounds.h, opacity);

        // Engraved detent marks down the left of the track.
        let last = (self.positions - 1) as f32;
        let mut detents = vg::Path::new();
        for index in 0..self.positions {
            let travel = 1.0 - index as f32 / last;
            let y = bounds.y + bounds.h * (0.12 + 0.76 * travel);
            detents.move_to(bounds.x + bounds.w * 0.08, y);
            detents.line_to(bounds.x + bounds.w * 0.26, y);
        }
        let mut detent_paint = vg::Paint::color(theme::with_opacity(theme::PANEL_ENGRAVE, opacity));
        detent_paint.set_line_width((bounds.w * 0.06).max(1.0));
        canvas.stroke_path(&detents, &detent_paint);

        let index = (self.normalized.get(cx).clamp(0.0, 1.0) * last).round();
        let travel = 1.0 - index / last;
        draw_lever(
            canvas,
            bounds.x + bounds.w * 0.5,
            bounds.y,
            bounds.h,
            bounds.w,
            travel,
            opacity,
        );
    }
}

/// The black phenolic plate a toggle is mounted through.
fn draw_base_plate(canvas: &mut Canvas, x: f32, y: f32, width: f32, height: f32, opacity: f32) {
    let inset = width * 0.18;
    let mut plate = vg::Path::new();
    plate.rounded_rect(
        x + inset,
        y + height * 0.06,
        width - inset * 2.0,
        height * 0.88,
        width * 0.22,
    );
    canvas.fill_path(
        &plate,
        &vg::Paint::color(theme::with_opacity(theme::HARDWARE_BASE, opacity)),
    );

    let mut bezel = vg::Path::new();
    bezel.rounded_rect(
        x + inset,
        y + height * 0.06,
        width - inset * 2.0,
        height * 0.88,
        width * 0.22,
    );
    let mut bezel_paint = vg::Paint::color(theme::with_opacity(theme::CHROME_DARK, opacity));
    bezel_paint.set_line_width((width * 0.05).max(0.8));
    canvas.stroke_path(&bezel, &bezel_paint);
}

/// The chrome bat lever itself.
///
/// `travel` is `0.0` at the top of the track and `1.0` at the bottom. The lever
/// is drawn as a tapered quadrilateral with a ball tip, shaded by a horizontal
/// gradient so it reads as a turned metal cylinder.
fn draw_lever(
    canvas: &mut Canvas,
    centre_x: f32,
    top: f32,
    height: f32,
    width: f32,
    travel: f32,
    opacity: f32,
) {
    // The pivot sits at the centre of the plate; the tip swings between the
    // upper and lower detents.
    let pivot_y = top + height * 0.5;
    let tip_y = top + height * (LEVER_TRAVEL_TOP + LEVER_TRAVEL_SPAN * travel);
    let base_half = width * 0.16;
    let tip_half = width * 0.11;

    let mut lever = vg::Path::new();
    lever.move_to(centre_x - base_half, pivot_y);
    lever.line_to(centre_x - tip_half, tip_y);
    lever.line_to(centre_x + tip_half, tip_y);
    lever.line_to(centre_x + base_half, pivot_y);
    lever.close();

    let paint = vg::Paint::linear_gradient(
        centre_x - base_half,
        pivot_y,
        centre_x + base_half,
        pivot_y,
        theme::with_opacity(theme::CHROME_LIGHT, opacity),
        theme::with_opacity(theme::CHROME_DARK, opacity),
    );
    canvas.fill_path(&lever, &paint);

    let mut tip = vg::Path::new();
    tip.circle(centre_x, tip_y, width * 0.155);
    let tip_paint = vg::Paint::radial_gradient(
        centre_x - width * 0.05,
        tip_y - width * 0.05,
        width * 0.02,
        width * 0.20,
        theme::with_opacity(theme::CHROME_LIGHT, opacity),
        theme::with_opacity(theme::CHROME_DARK, opacity),
    );
    canvas.fill_path(&tip, &tip_paint);

    // Collar around the pivot, so the lever appears to pass through the plate.
    let mut collar = vg::Path::new();
    collar.circle(centre_x, pivot_y, width * 0.20);
    let collar_paint = vg::Paint::linear_gradient(
        centre_x,
        pivot_y - width * 0.2,
        centre_x,
        pivot_y + width * 0.2,
        theme::with_opacity(theme::CHROME_DARK, opacity),
        theme::with_opacity(theme::HARDWARE_BASE, opacity),
    );
    canvas.fill_path(&collar, &collar_paint);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The detent-selection arithmetic, mirrored so it can be checked without a
    /// live event context.
    fn index_for(cursor_fraction: f32, positions: usize) -> f32 {
        let last = (positions - 1) as f32;
        ((1.0 - cursor_fraction.clamp(0.0, 1.0)) * last)
            .round()
            .clamp(0.0, last)
    }

    #[test]
    fn clicking_the_top_selects_the_last_variant() {
        // Power switch: Off, Standby, On. Clicking the top must select On.
        assert_eq!(index_for(0.0, 3), 2.0);
        assert_eq!(index_for(0.5, 3), 1.0);
        assert_eq!(index_for(1.0, 3), 0.0);
    }

    #[test]
    fn four_position_lever_addresses_every_wattage_mode() {
        // 100 W, 70 W, 50 W, 30 W: four detents, top is index 3.
        assert_eq!(index_for(0.0, 4), 3.0);
        assert_eq!(index_for(0.34, 4), 2.0);
        assert_eq!(index_for(0.66, 4), 1.0);
        assert_eq!(index_for(1.0, 4), 0.0);
    }

    #[test]
    fn cursor_positions_outside_the_widget_are_clamped() {
        for positions in 2..=4 {
            let last = (positions - 1) as f32;
            assert_eq!(index_for(-5.0, positions), last);
            assert_eq!(index_for(5.0, positions), 0.0);
        }
    }

    #[test]
    fn every_detent_maps_back_to_a_distinct_normalized_value() {
        for positions in 2..=4 {
            let last = (positions - 1) as f32;
            let mut seen = Vec::new();
            for step in 0..=100 {
                let normalized = index_for(step as f32 / 100.0, positions) / last;
                assert!((0.0..=1.0).contains(&normalized));
                if !seen.contains(&normalized.to_bits()) {
                    seen.push(normalized.to_bits());
                }
            }
            assert_eq!(seen.len(), positions, "missed a detent at {positions}");
        }
    }

    #[test]
    fn lever_travel_is_bounded_within_its_plate() {
        // Sweep the whole travel range, including every detent each switch
        // size can select, and check the tip stays on the plate.
        let mut lowest = f32::INFINITY;
        let mut highest = f32::NEG_INFINITY;
        for step in 0..=100 {
            let travel = step as f32 / 100.0;
            let tip = LEVER_TRAVEL_TOP + LEVER_TRAVEL_SPAN * travel;
            assert!(
                (0.0..=1.0).contains(&tip),
                "lever tip left the plate at {travel}"
            );
            lowest = lowest.min(tip);
            highest = highest.max(tip);
        }
        // The travel must actually be a travel, not a fixed position.
        assert!(highest - lowest > 0.5, "lever barely moves");
    }
}
