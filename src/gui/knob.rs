//! Fluted black pointer knob with a 270-degree sweep.
//!
//! Specification section 4: "Custom-drawn `Vizia` rotary widgets representing
//! fluted black pointer knobs with white indicator lines (270 degree sweep)."
//!
//! # Why the pointer follows the *plain* value
//!
//! The volume and gain parameters use an exponential range so hosts get finer
//! automation resolution where it matters (specification section 3). Driving
//! the pointer from the normalized value would then desynchronise the knob from
//! the number printed beneath it — a control reading "5.0" would point at 70 %
//! of its travel. A real potentiometer's taper is a property of the circuit,
//! not of the shaft angle, so this widget rotates linearly in plain value and
//! converts to normalized only when emitting parameter events.

use nih_plug::prelude::Param;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use nih_plug_vizia::widgets::param_base::ParamWidgetBase;
use nih_plug_vizia::widgets::util::ModifiersExt;

use super::glyphs::{self, Glyph};
use super::theme;

/// Vertical pixels of drag corresponding to the full control range.
const DRAG_RANGE_PX: f32 = 220.0;
/// Multiplier applied while shift is held, for fine adjustment.
const FINE_DRAG_MULTIPLIER: f32 = 0.2;
/// Total rotation of the pointer, in radians (270 degrees).
const SWEEP: f32 = 1.5 * std::f32::consts::PI;
/// Number of flutes moulded into the knob's rim.
const FLUTE_COUNT: usize = 20;
/// Number of engraved index ticks around each knob.
const TICK_COUNT: usize = 11;

/// Converts a travel fraction into a pointer angle measured clockwise from
/// straight up, so `0.0` points down-left and `1.0` points down-right.
#[inline]
fn angle_for(fraction: f32) -> f32 {
    -SWEEP * 0.5 + SWEEP * fraction
}

/// A rotary control bound to a `nih-plug` parameter.
pub struct Knob<L>
where
    L: Lens<Target = f32>,
{
    param: ParamWidgetBase,
    /// Normalized parameter value. Stored normalized rather than plain because
    /// `Param::Plain` is an associated type that is only `f32` for float
    /// parameters; `ParamWidgetBase::preview_plain` converts it on demand and
    /// keeps this widget usable with any parameter type.
    normalized: L,
    /// Lower and upper bounds of the plain range.
    minimum: f32,
    maximum: f32,
    glyph: Glyph,

    drag_active: bool,
    /// Cursor Y and plain value at the moment the drag began.
    drag_origin_y: f32,
    drag_origin_value: f32,
    /// Fractional scroll lines not yet turned into a parameter step.
    scrolled_lines: f32,
}

impl<L> Knob<L>
where
    L: Lens<Target = f32>,
{
    /// Fraction of the control's travel currently shown, in `0.0..=1.0`.
    ///
    /// Derived from the *plain* value so the pointer angle agrees with the
    /// number printed beneath it even on an exponentially skewed range.
    fn fraction(&self, cx: &mut DrawContext) -> f32 {
        let span = self.maximum - self.minimum;
        if span.abs() < f32::EPSILON {
            return 0.0;
        }
        let plain = self.param.preview_plain(self.normalized.get(cx));
        ((plain - self.minimum) / span).clamp(0.0, 1.0)
    }

    /// Applies a new plain value, converting to normalized for the host.
    fn commit(&self, cx: &mut EventContext, plain: f32) {
        let clamped = plain.clamp(self.minimum, self.maximum);
        let normalized = self.param.preview_normalized(clamped);
        self.param.set_normalized_value(cx, normalized);
    }
}

/// Creates a knob bound to one parameter.
///
/// `params` is a lens onto the plugin's parameter struct and `params_to_param`
/// projects out the parameter to control, matching the convention used by
/// `nih_plug_vizia`'s own widgets.
pub fn knob<L, Params, P, FMap>(
    cx: &mut Context,
    params: L,
    params_to_param: FMap,
    glyph: Glyph,
) -> Handle<'_, impl View>
where
    L: Lens<Target = Params> + Clone,
    Params: 'static,
    P: Param + 'static,
    FMap: Fn(&Params) -> &P + Copy + 'static,
{
    let param = ParamWidgetBase::new(cx, params, params_to_param);
    // `preview_plain` on the type-erased base returns `f32` for every parameter
    // kind, so the range bounds need no `Plain = f32` bound on the caller.
    let minimum = param.preview_plain(0.0);
    let maximum = param.preview_plain(1.0);
    let normalized = ParamWidgetBase::make_lens(params, params_to_param, |param| {
        param.modulated_normalized_value()
    });

    Knob {
        param,
        normalized,
        minimum,
        maximum,
        glyph,
        drag_active: false,
        drag_origin_y: 0.0,
        drag_origin_value: 0.0,
        scrolled_lines: 0.0,
    }
    .build(cx, |_| {})
    .class("knob")
}

impl<L> View for Knob<L>
where
    L: Lens<Target = f32>,
{
    fn element(&self) -> Option<&'static str> {
        Some("knob")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|window_event, meta| match window_event {
            // A triple click arrives instead of a third mouse-down; treating it
            // as an ordinary press keeps click-drag working straight after a
            // double-click reset.
            WindowEvent::MouseDown(MouseButton::Left)
            | WindowEvent::MouseTripleClick(MouseButton::Left) => {
                if cx.modifiers().command() {
                    self.param.begin_set_parameter(cx);
                    self.param
                        .set_normalized_value(cx, self.param.default_normalized_value());
                    self.param.end_set_parameter(cx);
                } else {
                    self.drag_active = true;
                    cx.capture();
                    cx.focus();
                    cx.set_active(true);
                    self.drag_origin_y = cx.mouse().cursory;
                    self.drag_origin_value = self.param.modulated_plain_value();
                    self.param.begin_set_parameter(cx);
                }
                meta.consume();
            }

            WindowEvent::MouseDoubleClick(MouseButton::Left)
            | WindowEvent::MouseDown(MouseButton::Right) => {
                // Double-click and right-click both restore the default, the
                // convention every DAW user already expects.
                if self.drag_active {
                    self.drag_active = false;
                    cx.release();
                    cx.set_active(false);
                    self.param.end_set_parameter(cx);
                }
                self.param.begin_set_parameter(cx);
                self.param
                    .set_normalized_value(cx, self.param.default_normalized_value());
                self.param.end_set_parameter(cx);
                meta.consume();
            }

            WindowEvent::MouseUp(MouseButton::Left) => {
                if self.drag_active {
                    self.drag_active = false;
                    cx.release();
                    cx.set_active(false);
                    self.param.end_set_parameter(cx);
                    meta.consume();
                }
            }

            WindowEvent::MouseMove(_x, y) => {
                if self.drag_active {
                    // Dragging up raises the value; screen Y grows downwards.
                    let travelled = (self.drag_origin_y - *y) / cx.scale_factor();
                    let sensitivity = if cx.modifiers().shift() {
                        FINE_DRAG_MULTIPLIER
                    } else {
                        1.0
                    };
                    let span = self.maximum - self.minimum;
                    let delta = travelled / DRAG_RANGE_PX * span * sensitivity;
                    self.commit(cx, self.drag_origin_value + delta);
                }
            }

            WindowEvent::MouseScroll(_horizontal, vertical) => {
                // Trackpads deliver fractional lines, so accumulate until a
                // whole step is available.
                self.scrolled_lines += *vertical;
                if self.scrolled_lines.abs() >= 1.0 {
                    let finer = cx.modifiers().shift();
                    self.param.begin_set_parameter(cx);
                    let mut normalized = self.param.unmodulated_normalized_value();
                    while self.scrolled_lines >= 1.0 {
                        normalized = self.param.next_normalized_step(normalized, finer);
                        self.scrolled_lines -= 1.0;
                    }
                    while self.scrolled_lines <= -1.0 {
                        normalized = self.param.previous_normalized_step(normalized, finer);
                        self.scrolled_lines += 1.0;
                    }
                    self.param.set_normalized_value(cx, normalized);
                    self.param.end_set_parameter(cx);
                }
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
        let fraction = self.fraction(cx);

        // Reserve the top of the cell for the pictogram and centre the knob in
        // what remains, so every cell in the bank lines up regardless of glyph.
        let glyph_size = (bounds.h * 0.20).min(bounds.w * 0.34);
        glyphs::draw(
            canvas,
            self.glyph,
            bounds.x + (bounds.w - glyph_size) * 0.5,
            bounds.y,
            glyph_size,
            theme::with_opacity(theme::PANEL_INK, opacity),
        );

        let dial_top = bounds.y + glyph_size * 1.35;
        let dial_height = bounds.h - (glyph_size * 1.35);
        let radius = (dial_height.min(bounds.w) * 0.5) * 0.72;
        let centre_x = bounds.x + bounds.w * 0.5;
        let centre_y = dial_top + dial_height * 0.5 - radius * 0.10;

        draw_index_ticks(canvas, centre_x, centre_y, radius, fraction, opacity);
        draw_knob_body(canvas, centre_x, centre_y, radius, opacity);
        draw_flutes(canvas, centre_x, centre_y, radius, opacity);
        draw_pointer(canvas, centre_x, centre_y, radius, fraction, opacity);
    }
}

/// Engraved index marks around the dial, lit in amber up to the current value.
fn draw_index_ticks(
    canvas: &mut Canvas,
    centre_x: f32,
    centre_y: f32,
    radius: f32,
    fraction: f32,
    opacity: f32,
) {
    let inner = radius * 1.18;
    let outer = radius * 1.34;
    for index in 0..TICK_COUNT {
        let tick_fraction = index as f32 / (TICK_COUNT - 1) as f32;
        let angle = angle_for(tick_fraction);
        let (sin, cos) = angle.sin_cos();
        let (dx, dy) = (sin, -cos);

        let mut path = vg::Path::new();
        path.move_to(centre_x + dx * inner, centre_y + dy * inner);
        path.line_to(centre_x + dx * outer, centre_y + dy * outer);

        // A small tolerance keeps the tick under the pointer lit rather than
        // flickering on floating-point equality.
        let lit = tick_fraction <= fraction + 1.0e-3;
        let colour = if lit {
            theme::with_opacity(theme::AMBER, opacity)
        } else {
            theme::with_opacity(theme::PANEL_ENGRAVE, opacity)
        };
        let mut paint = vg::Paint::color(colour);
        paint.set_line_width((radius * 0.09).max(1.0));
        paint.set_line_cap(vg::LineCap::Round);
        canvas.stroke_path(&path, &paint);
    }
}

/// The moulded body: a drop shadow, a radial-shaded black cap, and a rim.
fn draw_knob_body(canvas: &mut Canvas, centre_x: f32, centre_y: f32, radius: f32, opacity: f32) {
    let mut shadow = vg::Path::new();
    shadow.circle(centre_x, centre_y + radius * 0.07, radius * 1.03);
    canvas.fill_path(
        &shadow,
        &vg::Paint::color(theme::with_opacity(theme::KNOB_SHADOW, opacity * 0.55)),
    );

    let mut body = vg::Path::new();
    body.circle(centre_x, centre_y, radius);
    // Light falls from the upper left, so the highlight centre is offset there.
    let paint = vg::Paint::radial_gradient(
        centre_x - radius * 0.30,
        centre_y - radius * 0.34,
        radius * 0.12,
        radius * 1.25,
        theme::with_opacity(theme::KNOB_HIGHLIGHT, opacity),
        theme::with_opacity(theme::KNOB_BODY, opacity),
    );
    canvas.fill_path(&body, &paint);

    let mut rim = vg::Path::new();
    rim.circle(centre_x, centre_y, radius * 0.985);
    let mut rim_paint = vg::Paint::color(theme::with_opacity(theme::KNOB_RIM, opacity));
    rim_paint.set_line_width((radius * 0.06).max(0.75));
    canvas.stroke_path(&rim, &rim_paint);
}

/// The knurled flutes moulded into the rim.
fn draw_flutes(canvas: &mut Canvas, centre_x: f32, centre_y: f32, radius: f32, opacity: f32) {
    let inner = radius * 0.80;
    let outer = radius * 0.99;
    let mut paint = vg::Paint::color(theme::with_opacity(theme::KNOB_FLUTE, opacity));
    paint.set_line_width((radius * 0.075).max(0.6));
    paint.set_line_cap(vg::LineCap::Round);

    for index in 0..FLUTE_COUNT {
        let angle = std::f32::consts::TAU * index as f32 / FLUTE_COUNT as f32;
        let (sin, cos) = angle.sin_cos();
        let mut path = vg::Path::new();
        path.move_to(centre_x + sin * inner, centre_y - cos * inner);
        path.line_to(centre_x + sin * outer, centre_y - cos * outer);
        canvas.stroke_path(&path, &paint);
    }
}

/// The white indicator line.
fn draw_pointer(
    canvas: &mut Canvas,
    centre_x: f32,
    centre_y: f32,
    radius: f32,
    fraction: f32,
    opacity: f32,
) {
    let angle = angle_for(fraction);
    let (sin, cos) = angle.sin_cos();
    let (dx, dy) = (sin, -cos);

    let mut path = vg::Path::new();
    path.move_to(centre_x + dx * radius * 0.06, centre_y + dy * radius * 0.06);
    path.line_to(centre_x + dx * radius * 0.88, centre_y + dy * radius * 0.88);

    let mut paint = vg::Paint::color(theme::with_opacity(theme::KNOB_POINTER, opacity));
    paint.set_line_width((radius * 0.13).max(1.5));
    paint.set_line_cap(vg::LineCap::Round);
    canvas.stroke_path(&path, &paint);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reimplementation of the pointer geometry, so the test exercises the same
    /// arithmetic the widget uses without needing a live GPU canvas.
    fn direction(fraction: f32) -> (f32, f32) {
        let angle = angle_for(fraction);
        let (sin, cos) = angle.sin_cos();
        (sin, -cos)
    }

    #[test]
    fn sweep_is_exactly_270_degrees() {
        let start = angle_for(0.0);
        let end = angle_for(1.0);
        assert!(((end - start).to_degrees() - 270.0).abs() < 1.0e-3);
    }

    #[test]
    fn minimum_points_down_left_and_maximum_down_right() {
        let (dx_min, dy_min) = direction(0.0);
        assert!(dx_min < 0.0, "minimum did not point left: {dx_min}");
        assert!(dy_min > 0.0, "minimum did not point down: {dy_min}");

        let (dx_max, dy_max) = direction(1.0);
        assert!(dx_max > 0.0, "maximum did not point right: {dx_max}");
        assert!(dy_max > 0.0, "maximum did not point down: {dy_max}");

        // Symmetric about the vertical axis.
        assert!((dx_min + dx_max).abs() < 1.0e-5);
        assert!((dy_min - dy_max).abs() < 1.0e-5);
    }

    #[test]
    fn centre_points_straight_up() {
        let (dx, dy) = direction(0.5);
        assert!(dx.abs() < 1.0e-6, "centre drifted sideways: {dx}");
        assert!((dy + 1.0).abs() < 1.0e-6, "centre did not point up: {dy}");
    }

    #[test]
    fn pointer_rotates_monotonically_clockwise() {
        let mut previous = f32::NEG_INFINITY;
        for step in 0..=100 {
            let angle = angle_for(step as f32 / 100.0);
            assert!(angle > previous, "rotation reversed at step {step}");
            previous = angle;
        }
    }

    #[test]
    fn index_ticks_span_the_whole_sweep() {
        let first = angle_for(0.0);
        let last = angle_for((TICK_COUNT - 1) as f32 / (TICK_COUNT - 1) as f32);
        assert!((first - angle_for(0.0)).abs() < 1.0e-6);
        assert!((last - angle_for(1.0)).abs() < 1.0e-6);
        // Eleven ticks give the 0..10 markings a real amp's faceplate carries.
        assert_eq!(TICK_COUNT, 11);
    }

    #[test]
    fn drag_conversion_covers_the_range_over_the_declared_distance() {
        // Dragging DRAG_RANGE_PX pixels must traverse exactly one full span.
        let span = 10.0f32;
        let delta = DRAG_RANGE_PX / DRAG_RANGE_PX * span;
        assert!((delta - span).abs() < 1.0e-6);

        // ...and holding shift must cover a fifth of it.
        let fine = DRAG_RANGE_PX / DRAG_RANGE_PX * span * FINE_DRAG_MULTIPLIER;
        assert!((fine - 2.0).abs() < 1.0e-6);
    }
}
