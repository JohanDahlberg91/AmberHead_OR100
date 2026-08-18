//! The amplifier head itself: tolex, corners, chrome and the control panel.
//!
//! Specification section 4 asks for a skeuomorphic faceplate. This module draws
//! the physical object an OR100-style head actually is — a vinyl-covered
//! plywood box with moulded corner protectors, a chrome-framed aperture, and a
//! white control panel carrying a black-outlined bar of coloured cells — rather
//! than a flat rectangle with decorative stripes.
//!
//! # One source of truth for the layout
//!
//! Everything is expressed as a [`Plate`] in a 920x340 *logical design space*
//! and mapped onto the widget's real bounds at draw time. The `vizia`
//! stylesheet positions the interactive controls with the same numbers, so the
//! knobs land exactly inside the cells painted here. The constants below are
//! the authority; `theme.css` mirrors them and this module's tests check that
//! the cells tile the bar exactly and stay inside the panel, which is the part a
//! hand-edited stylesheet is most likely to break.
//!
//! # About the artwork
//!
//! The proportions, materials and colour scheme are those of the real
//! amplifier. The wordmark, the model plate's typeface and the badge are
//! **original**: Orange Amplification's logo and coat of arms are that
//! company's trademarks and are not reproduced here, in vector or otherwise.

use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;

use super::theme;
use super::{WINDOW_HEIGHT, WINDOW_WIDTH};

/// A rectangle in the faceplate's 920x340 logical design space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plate {
    /// Left edge, in logical pixels.
    pub x: f32,
    /// Top edge, in logical pixels.
    pub y: f32,
    /// Width, in logical pixels.
    pub w: f32,
    /// Height, in logical pixels.
    pub h: f32,
}

impl Plate {
    /// Builds a plate from its logical-space edges and size.
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Right edge, in logical pixels.
    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    /// Bottom edge, in logical pixels.
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    /// Whether `self` lies entirely inside `outer`.
    pub fn inside(&self, outer: &Plate) -> bool {
        self.x >= outer.x
            && self.y >= outer.y
            && self.right() <= outer.right()
            && self.bottom() <= outer.bottom()
    }

    /// Maps this plate onto a widget's on-screen bounds.
    ///
    /// The editor keeps a fixed logical size and scales through the host's
    /// scale factor, so the two axes normally share a factor; they are computed
    /// separately anyway so an unexpected aspect ratio stretches the chassis
    /// rather than tearing the panel off its background.
    pub fn on(&self, bounds: &BoundingBox) -> (f32, f32, f32, f32) {
        let scale_x = bounds.w / WINDOW_WIDTH as f32;
        let scale_y = bounds.h / WINDOW_HEIGHT as f32;
        (
            bounds.x + self.x * scale_x,
            bounds.y + self.y * scale_y,
            self.w * scale_x,
            self.h * scale_y,
        )
    }
}

/// The whole chassis.
pub const CHASSIS: Plate = Plate::new(0.0, 0.0, WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32);
/// The chrome-framed aperture and the white control panel inside it.
pub const PANEL: Plate = Plate::new(46.0, 58.0, 828.0, 220.0);
/// Header strip carrying the wordmark, the model plate and the badge.
pub const HEADER: Plate = Plate::new(70.0, 72.0, 780.0, 44.0);
/// The black-outlined control bar.
pub const BAR: Plate = Plate::new(64.0, 130.0, 792.0, 130.0);
/// Left cell: the pilot jewel and the six switches. Painted panel-white, as the
/// switch bank is on the real amplifier.
pub const CELL_SWITCHES: Plate = Plate::new(64.0, 130.0, 232.0, 130.0);
/// Middle cell: the clean channel's three controls, on orange.
pub const CELL_CLEAN: Plate = Plate::new(296.0, 130.0, 216.0, 130.0);
/// Right cell: the dirty channel's five controls, on orange.
pub const CELL_DIRTY: Plate = Plate::new(512.0, 130.0, 344.0, 130.0);

/// Corner radius of the tolex-covered box, in logical pixels.
const CHASSIS_RADIUS: f32 = 13.0;
/// Side of the moulded corner protectors, in logical pixels.
///
/// Bounded by the panel: the protectors wrap the extreme corners of the box and
/// must clear the chrome bezel, which
/// `tests::corner_protectors_and_the_panel_do_not_collide` checks.
const CORNER_SIZE: f32 = 52.0;
/// Corner radius of the panel aperture.
const PANEL_RADIUS: f32 = 24.0;
/// Width of the chrome trim around that aperture.
const CHROME_WIDTH: f32 = 5.0;
/// Corner radius of the control bar and its cells.
const BAR_RADIUS: f32 = 4.0;
/// Width of the black outline drawn around the bar and between its cells.
const BAR_INK_WIDTH: f32 = 3.0;

/// Speckles in the tolex grain. Enough to read as vinyl at 200 % scale without
/// turning the box into noise at 75 %.
const GRAIN_SPECKLES: usize = 5_200;
/// Seed of the grain generator. Fixed so the texture is identical on every
/// redraw and every instance: a pattern that shimmered between frames would be
/// far more distracting than no texture at all.
const GRAIN_SEED: u32 = 0x9E37_79B9;

/// The amplifier head. Draws everything that is not an interactive control.
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

        draw_tolex(canvas, &bounds, opacity);
        draw_grain(canvas, &bounds, opacity);
        draw_corner_protectors(canvas, &bounds, opacity);
        draw_panel(canvas, &bounds, opacity);
        draw_control_bar(canvas, &bounds, opacity);
    }
}

/// The vinyl-covered box: a rounded rectangle lit from above.
fn draw_tolex(canvas: &mut Canvas, bounds: &BoundingBox, opacity: f32) {
    let (x, y, w, h) = CHASSIS.on(bounds);
    let radius = scaled(CHASSIS_RADIUS, bounds);

    let mut box_path = vg::Path::new();
    box_path.rounded_rect(x, y, w, h, radius);
    canvas.fill_path(
        &box_path,
        &vg::Paint::linear_gradient(
            x,
            y,
            x,
            y + h,
            theme::with_opacity(theme::TOLEX_LIT, opacity),
            theme::with_opacity(theme::TOLEX_SHADE, opacity),
        ),
    );

    // A soft edge shadow all round, which is what stops the box reading as a
    // flat orange rectangle.
    let mut edge = vg::Path::new();
    edge.rounded_rect(x, y, w, h, radius);
    let mut edge_paint = vg::Paint::color(theme::with_opacity(theme::TOLEX_EDGE, opacity * 0.9));
    edge_paint.set_line_width(scaled(3.0, bounds));
    canvas.stroke_path(&edge, &edge_paint);
}

/// Pebbled vinyl grain.
///
/// Positions come from a fixed-seed linear congruential generator rather than a
/// random source, so the texture is reproducible; see [`GRAIN_SEED`].
fn draw_grain(canvas: &mut Canvas, bounds: &BoundingBox, opacity: f32) {
    let mut state = GRAIN_SEED;
    let mut next = || {
        // Numerical Recipes LCG constants.
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / 16_777_216.0
    };

    let dot = scaled(1.05, bounds);
    let mut dark = vg::Path::new();
    let mut light = vg::Path::new();
    for index in 0..GRAIN_SPECKLES {
        let x = bounds.x + next() * bounds.w;
        let y = bounds.y + next() * bounds.h;
        let size = dot * (0.5 + next());
        // Alternating the two paths gives the grain both its shadow and its
        // highlight, which a single-colour speckle cannot do.
        if index % 2 == 0 {
            dark.rect(x, y, size, size);
        } else {
            light.rect(x, y, size, size * 0.7);
        }
    }
    canvas.fill_path(
        &dark,
        &vg::Paint::color(theme::with_opacity(theme::TOLEX_GRAIN, opacity * 0.20)),
    );
    canvas.fill_path(
        &light,
        &vg::Paint::color(theme::with_opacity(theme::TOLEX_LIT, opacity * 0.22)),
    );
}

/// The four moulded plastic corner protectors.
fn draw_corner_protectors(canvas: &mut Canvas, bounds: &BoundingBox, opacity: f32) {
    let (x, y, w, h) = CHASSIS.on(bounds);
    let size_x = scaled(CORNER_SIZE, bounds);
    let size_y = scaled_y(CORNER_SIZE, bounds);
    let radius = scaled(CHASSIS_RADIUS, bounds);

    // (corner x, corner y, x direction, y direction) for each of the four.
    let corners = [
        (x, y, 1.0f32, 1.0f32),
        (x + w, y, -1.0, 1.0),
        (x, y + h, 1.0, -1.0),
        (x + w, y + h, -1.0, -1.0),
    ];

    for (corner_x, corner_y, dx, dy) in corners {
        let mut path = vg::Path::new();
        // Along the top edge, in from the corner...
        path.move_to(corner_x + dx * size_x, corner_y);
        path.line_to(corner_x + dx * radius, corner_y);
        // ...round the box's own corner radius...
        path.quad_to(corner_x, corner_y, corner_x, corner_y + dy * radius);
        // ...back down the side...
        path.line_to(corner_x, corner_y + dy * size_y);
        // ...and back across the inner face. The control points must sit
        // *beyond* the chord from (0, size) to (size, 0) — their coordinates
        // summing to more than `size` — or the curve collapses onto that chord
        // and the protector renders as a bare triangle. 0.42 + 0.88 gives the
        // moulding the slight belly a real corner cap has.
        path.bezier_to(
            corner_x + dx * size_x * 0.42,
            corner_y + dy * size_y * 0.88,
            corner_x + dx * size_x * 0.88,
            corner_y + dy * size_y * 0.42,
            corner_x + dx * size_x,
            corner_y,
        );
        path.close();

        canvas.fill_path(
            &path,
            &vg::Paint::linear_gradient(
                corner_x,
                corner_y,
                corner_x + dx * size_x,
                corner_y + dy * size_y,
                theme::with_opacity(theme::CORNER_LIT, opacity),
                theme::with_opacity(theme::CORNER_BODY, opacity),
            ),
        );

        let mut outline = vg::Paint::color(theme::with_opacity(theme::CORNER_EDGE, opacity));
        outline.set_line_width(scaled(1.2, bounds));
        canvas.stroke_path(&path, &outline);
    }
}

/// The chrome-framed aperture and the white control panel inside it.
fn draw_panel(canvas: &mut Canvas, bounds: &BoundingBox, opacity: f32) {
    let (x, y, w, h) = PANEL.on(bounds);
    let radius = scaled(PANEL_RADIUS, bounds);
    let chrome = scaled(CHROME_WIDTH, bounds);

    // Recess shadow, offset downwards so the panel reads as set into the box.
    let mut recess = vg::Path::new();
    recess.rounded_rect(
        x - chrome,
        y - chrome * 0.4,
        w + chrome * 2.0,
        h + chrome * 2.0,
        radius + chrome,
    );
    canvas.fill_path(
        &recess,
        &vg::Paint::color(theme::with_opacity(theme::TOLEX_EDGE, opacity * 0.55)),
    );

    // Chrome trim: a bright band across the top, dark along the bottom, which
    // is how a polished bezel catches a room light.
    let mut trim = vg::Path::new();
    trim.rounded_rect(
        x - chrome * 0.5,
        y - chrome * 0.5,
        w + chrome,
        h + chrome,
        radius + chrome * 0.5,
    );
    let mut trim_paint = vg::Paint::linear_gradient(
        x,
        y - chrome,
        x,
        y + h + chrome,
        theme::with_opacity(theme::CHROME_LIGHT, opacity),
        theme::with_opacity(theme::CHROME_DARK, opacity),
    );
    trim_paint.set_line_width(chrome);
    canvas.stroke_path(&trim, &trim_paint);

    // The panel itself.
    let mut panel = vg::Path::new();
    panel.rounded_rect(x, y, w, h, radius);
    canvas.fill_path(
        &panel,
        &vg::Paint::linear_gradient(
            x,
            y,
            x,
            y + h,
            theme::with_opacity(theme::PANEL_WHITE, opacity),
            theme::with_opacity(theme::PANEL_SHADE, opacity),
        ),
    );
}

/// The black-outlined control bar and its three coloured cells.
fn draw_control_bar(canvas: &mut Canvas, bounds: &BoundingBox, opacity: f32) {
    let radius = scaled(BAR_RADIUS, bounds);
    let ink = theme::with_opacity(theme::BAR_INK, opacity);

    for (cell, colour) in [
        (CELL_SWITCHES, theme::PANEL_WHITE),
        (CELL_CLEAN, theme::BAND_ORANGE),
        (CELL_DIRTY, theme::BAND_ORANGE),
    ] {
        let (x, y, w, h) = cell.on(bounds);
        let mut path = vg::Path::new();
        path.rounded_rect(x, y, w, h, radius);
        canvas.fill_path(
            &path,
            &vg::Paint::linear_gradient(
                x,
                y,
                x,
                y + h,
                theme::with_opacity(theme::mix(colour, theme::PANEL_WHITE, 0.12), opacity),
                theme::with_opacity(theme::mix(colour, theme::BAR_INK, 0.10), opacity),
            ),
        );
    }

    // One continuous outline around the bar, then a rule between each pair of
    // cells: drawing each cell's own border would double the line weight where
    // two cells meet.
    let (bar_x, bar_y, bar_w, bar_h) = BAR.on(bounds);
    let mut outline = vg::Path::new();
    outline.rounded_rect(bar_x, bar_y, bar_w, bar_h, radius);
    for cell in [CELL_SWITCHES, CELL_CLEAN] {
        let divider = Plate::new(cell.right(), cell.y, 0.0, cell.h);
        let (divider_x, divider_y, _, divider_h) = divider.on(bounds);
        outline.move_to(divider_x, divider_y);
        outline.line_to(divider_x, divider_y + divider_h);
    }
    let mut ink_paint = vg::Paint::color(ink);
    ink_paint.set_line_width(scaled(BAR_INK_WIDTH, bounds));
    canvas.stroke_path(&outline, &ink_paint);
}

/// The badge on the right of the header.
///
/// An original mark: a shield carrying the silhouette of a triode valve, which
/// is what the amplifier this models is full of. It is deliberately nothing
/// like Orange Amplification's coat of arms.
pub struct Crest;

impl View for Crest {
    fn element(&self) -> Option<&'static str> {
        Some("crest")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        if bounds.w <= 0.0 || bounds.h <= 0.0 {
            return;
        }
        let opacity = cx.opacity();
        let ink = theme::with_opacity(theme::BAR_INK, opacity);

        // Shield: square shoulders narrowing to a point.
        let (x, y, w, h) = (bounds.x, bounds.y, bounds.w, bounds.h);
        let mut shield = vg::Path::new();
        shield.move_to(x + w * 0.08, y + h * 0.06);
        shield.line_to(x + w * 0.92, y + h * 0.06);
        shield.line_to(x + w * 0.92, y + h * 0.58);
        shield.bezier_to(
            x + w * 0.92,
            y + h * 0.82,
            x + w * 0.66,
            y + h * 0.92,
            x + w * 0.50,
            y + h * 0.97,
        );
        shield.bezier_to(
            x + w * 0.34,
            y + h * 0.92,
            x + w * 0.08,
            y + h * 0.82,
            x + w * 0.08,
            y + h * 0.58,
        );
        shield.close();
        canvas.fill_path(
            &shield,
            &vg::Paint::color(theme::with_opacity(theme::PANEL_WHITE, opacity)),
        );
        let mut border = vg::Paint::color(ink);
        border.set_line_width((w * 0.055).max(1.0));
        canvas.stroke_path(&shield, &border);

        // Valve envelope.
        let glass_x = x + w * 0.5;
        let glass_top = y + h * 0.20;
        let glass_bottom = y + h * 0.62;
        let glass_radius = w * 0.17;
        let mut glass = vg::Path::new();
        glass.move_to(glass_x - glass_radius, glass_bottom);
        glass.line_to(glass_x - glass_radius, glass_top + glass_radius);
        glass.quad_to(glass_x - glass_radius, glass_top, glass_x, glass_top);
        glass.quad_to(
            glass_x + glass_radius,
            glass_top,
            glass_x + glass_radius,
            glass_top + glass_radius,
        );
        glass.line_to(glass_x + glass_radius, glass_bottom);
        glass.close();
        canvas.fill_path(
            &glass,
            &vg::Paint::color(theme::with_opacity(theme::AMBER, opacity * 0.55)),
        );
        let mut glass_edge = vg::Paint::color(ink);
        glass_edge.set_line_width((w * 0.045).max(0.8));
        canvas.stroke_path(&glass, &glass_edge);

        // Base and pins.
        let mut base = vg::Path::new();
        base.rect(
            glass_x - glass_radius * 0.95,
            glass_bottom,
            glass_radius * 1.9,
            h * 0.09,
        );
        canvas.fill_path(&base, &vg::Paint::color(ink));

        let mut pins = vg::Path::new();
        for step in -1..=1 {
            let pin_x = glass_x + step as f32 * glass_radius * 0.62;
            pins.move_to(pin_x, glass_bottom + h * 0.09);
            pins.line_to(pin_x, glass_bottom + h * 0.16);
        }
        let mut pin_paint = vg::Paint::color(ink);
        pin_paint.set_line_width((w * 0.05).max(0.8));
        pin_paint.set_line_cap(vg::LineCap::Round);
        canvas.stroke_path(&pins, &pin_paint);

        // Filament.
        let mut filament = vg::Path::new();
        filament.move_to(glass_x - glass_radius * 0.4, glass_bottom - h * 0.04);
        filament.line_to(glass_x, glass_top + h * 0.10);
        filament.line_to(glass_x + glass_radius * 0.4, glass_bottom - h * 0.04);
        let mut filament_paint = vg::Paint::color(theme::with_opacity(theme::AMBER_DEEP, opacity));
        filament_paint.set_line_width((w * 0.045).max(0.8));
        canvas.stroke_path(&filament, &filament_paint);
    }
}

/// Scales a logical-space length by the widget's horizontal scale.
fn scaled(length: f32, bounds: &BoundingBox) -> f32 {
    length * bounds.w / WINDOW_WIDTH as f32
}

/// Scales a logical-space length by the widget's vertical scale.
fn scaled_y(length: f32, bounds: &BoundingBox) -> f32 {
    length * bounds.h / WINDOW_HEIGHT as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds_at(scale: f32) -> BoundingBox {
        BoundingBox {
            x: 0.0,
            y: 0.0,
            w: WINDOW_WIDTH as f32 * scale,
            h: WINDOW_HEIGHT as f32 * scale,
        }
    }

    #[test]
    fn the_cells_tile_the_control_bar_exactly() {
        // Any gap or overlap here shows as a doubled or missing black rule
        // between two cells, and puts the knobs off their painted background.
        assert_eq!(CELL_SWITCHES.x, BAR.x);
        assert_eq!(CELL_SWITCHES.right(), CELL_CLEAN.x);
        assert_eq!(CELL_CLEAN.right(), CELL_DIRTY.x);
        assert_eq!(CELL_DIRTY.right(), BAR.right());

        let total = CELL_SWITCHES.w + CELL_CLEAN.w + CELL_DIRTY.w;
        assert_eq!(total, BAR.w, "the cells do not add up to the bar");

        for cell in [CELL_SWITCHES, CELL_CLEAN, CELL_DIRTY] {
            assert_eq!(cell.y, BAR.y);
            assert_eq!(cell.h, BAR.h);
        }
    }

    #[test]
    fn every_element_sits_inside_the_one_that_contains_it() {
        assert!(PANEL.inside(&CHASSIS), "the panel hangs off the box");
        assert!(HEADER.inside(&PANEL), "the header hangs off the panel");
        assert!(BAR.inside(&PANEL), "the control bar hangs off the panel");
        // The bar must clear the header rather than overlapping the wordmark.
        assert!(BAR.y > HEADER.bottom(), "the bar overlaps the header");
    }

    #[test]
    fn the_panel_is_inset_from_the_box_on_every_side() {
        // A panel flush with an edge would leave no tolex to read as a box,
        // and would collide with the corner protectors.
        let left = PANEL.x - CHASSIS.x;
        let right = CHASSIS.right() - PANEL.right();
        let top = PANEL.y - CHASSIS.y;
        let bottom = CHASSIS.bottom() - PANEL.bottom();
        for (name, margin) in [
            ("left", left),
            ("right", right),
            ("top", top),
            ("bottom", bottom),
        ] {
            assert!(margin >= CHASSIS_RADIUS, "{name} margin is only {margin}");
        }
        assert_eq!(left, right, "the panel is not centred horizontally");
        // Real heads sit the panel slightly above centre; the deeper skirt
        // below is where the chassis and valves live.
        assert!(bottom > top, "the panel is not sitting above centre");
    }

    #[test]
    fn every_knob_gets_a_legible_share_of_its_cell() {
        // A guitar amp knob needs about 60 logical pixels to stay legible with
        // its pictogram above it. Measured through `Plate::on`, which is the
        // path the cells are actually laid out by.
        const MIN_KNOB_WIDTH: f32 = 60.0;
        let bounds = bounds_at(1.0);
        for (name, cell, knobs) in [("clean", CELL_CLEAN, 3.0f32), ("dirty", CELL_DIRTY, 5.0)] {
            let (_, _, width, _) = cell.on(&bounds);
            let per_knob = width / knobs;
            assert!(
                per_knob >= MIN_KNOB_WIDTH,
                "{name} knobs would be only {per_knob} wide"
            );
        }
    }

    #[test]
    fn plates_map_onto_bounds_proportionally_at_every_scale() {
        for scale in [0.75f32, 1.0, 1.5, 2.0] {
            let bounds = bounds_at(scale);
            let (x, y, w, h) = PANEL.on(&bounds);
            assert!((x - PANEL.x * scale).abs() < 1.0e-3);
            assert!((y - PANEL.y * scale).abs() < 1.0e-3);
            assert!((w - PANEL.w * scale).abs() < 1.0e-3);
            assert!((h - PANEL.h * scale).abs() < 1.0e-3);
            // And it stays inside the widget at every scale.
            assert!(x >= bounds.x && x + w <= bounds.x + bounds.w);
            assert!(y >= bounds.y && y + h <= bounds.y + bounds.h);
        }
    }

    #[test]
    fn plates_are_offset_by_the_widgets_own_origin() {
        // The faceplate is not always at the window origin; a plate that
        // ignored `bounds.x` would paint the whole chassis in the wrong place.
        let bounds = BoundingBox {
            x: 37.0,
            y: 11.0,
            w: WINDOW_WIDTH as f32,
            h: WINDOW_HEIGHT as f32,
        };
        let (x, y, _, _) = BAR.on(&bounds);
        assert!((x - (37.0 + BAR.x)).abs() < 1.0e-3);
        assert!((y - (11.0 + BAR.y)).abs() < 1.0e-3);
    }

    /// Whether two plates share any area.
    fn overlaps(left: &Plate, right: &Plate) -> bool {
        left.x < right.right()
            && right.x < left.right()
            && left.y < right.bottom()
            && right.y < left.bottom()
    }

    #[test]
    fn corner_protectors_and_the_panel_do_not_collide() {
        // The protectors wrap the extreme corners of the box; the panel and its
        // chrome bezel are set into the middle. An overlap would draw moulded
        // plastic across the trim.
        //
        // Comparing only the horizontal extents is not enough and gives a false
        // failure: the protectors run down the *ends* of the box while the
        // panel is inset vertically, so the two clear each other in `y` even
        // though their `x` ranges intersect.
        let bezel = Plate::new(
            PANEL.x - CHROME_WIDTH,
            PANEL.y - CHROME_WIDTH,
            PANEL.w + CHROME_WIDTH * 2.0,
            PANEL.h + CHROME_WIDTH * 2.0,
        );

        for (name, x, y) in [
            ("top left", CHASSIS.x, CHASSIS.y),
            ("top right", CHASSIS.right() - CORNER_SIZE, CHASSIS.y),
            ("bottom left", CHASSIS.x, CHASSIS.bottom() - CORNER_SIZE),
            (
                "bottom right",
                CHASSIS.right() - CORNER_SIZE,
                CHASSIS.bottom() - CORNER_SIZE,
            ),
        ] {
            let protector = Plate::new(x, y, CORNER_SIZE, CORNER_SIZE);
            assert!(
                protector.inside(&CHASSIS),
                "the {name} protector hangs off the box"
            );
            assert!(
                !overlaps(&protector, &bezel),
                "the {name} protector overlaps the panel bezel"
            );
        }

        // The two protectors on one end must not run into each other.
        let top = Plate::new(CHASSIS.x, CHASSIS.y, CORNER_SIZE, CORNER_SIZE);
        let bottom = Plate::new(
            CHASSIS.x,
            CHASSIS.bottom() - CORNER_SIZE,
            CORNER_SIZE,
            CORNER_SIZE,
        );
        assert!(
            !overlaps(&top, &bottom),
            "the protectors meet in the middle of the end panel"
        );
    }

    #[test]
    fn the_grain_generator_is_reproducible_and_spread() {
        let sample = || {
            let mut state = GRAIN_SEED;
            let mut values = Vec::with_capacity(GRAIN_SPECKLES);
            for _ in 0..GRAIN_SPECKLES {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                values.push((state >> 8) as f32 / 16_777_216.0);
            }
            values
        };
        let first = sample();
        assert_eq!(first, sample(), "the grain is not reproducible");
        assert!(first.iter().all(|value| (0.0..1.0).contains(value)));

        // A degenerate generator that returned a constant would pile every
        // speckle in one spot.
        let mean: f32 = first.iter().sum::<f32>() / first.len() as f32;
        assert!((0.45..0.55).contains(&mean), "grain is not spread: {mean}");
    }

    #[test]
    fn scaling_helpers_follow_their_own_axis() {
        // 920x340 is not square, so a length scaled by the wrong axis is
        // visibly wrong; this is what keeps the corner protectors square-ish.
        let bounds = BoundingBox {
            x: 0.0,
            y: 0.0,
            w: WINDOW_WIDTH as f32 * 2.0,
            h: WINDOW_HEIGHT as f32 * 3.0,
        };
        assert!((scaled(10.0, &bounds) - 20.0).abs() < 1.0e-4);
        assert!((scaled_y(10.0, &bounds) - 30.0).abs() < 1.0e-4);
    }
}
