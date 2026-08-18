//! Faceplate palette.
//!
//! Specification section 4: "fluted black pointer knobs with white indicator
//! lines", "Dynamic amber pilot light", drawn on the amplifier the plugin
//! models — an orange vinyl-covered box with black corner protectors, a
//! chrome-framed aperture and a white control panel carrying a black-outlined
//! bar of coloured cells.
//!
//! Every colour the custom-drawn widgets use lives here so the whole faceplate
//! can be revoiced from one place, and so the CSS-styled containers and the
//! canvas-drawn controls cannot drift apart.

use nih_plug_vizia::vizia::vg;

/// Builds an opaque colour from 8-bit channels.
const fn rgb(red: u8, green: u8, blue: u8) -> vg::Color {
    vg::Color {
        r: red as f32 / 255.0,
        g: green as f32 / 255.0,
        b: blue as f32 / 255.0,
        a: 1.0,
    }
}

/// The amplifier's signature orange, used for the control-bar cells.
pub const ORANGE: vg::Color = rgb(0xE8, 0x62, 0x0E);
/// Deeper orange for shadowed orange surfaces and printed accents.
pub const ORANGE_DEEP: vg::Color = rgb(0xB4, 0x46, 0x06);
/// Fill of the two orange control-bar cells.
pub const BAND_ORANGE: vg::Color = ORANGE;

/// Vinyl covering where the light falls, at the top of the box.
pub const TOLEX_LIT: vg::Color = rgb(0xF0, 0x6E, 0x14);
/// The same vinyl in shadow, at the bottom.
pub const TOLEX_SHADE: vg::Color = rgb(0xCE, 0x55, 0x08);
/// Dark half of the pebbled grain in the vinyl.
pub const TOLEX_GRAIN: vg::Color = rgb(0x8A, 0x38, 0x04);
/// Shadow along the box's edges and in the panel recess.
pub const TOLEX_EDGE: vg::Color = rgb(0x6E, 0x2C, 0x03);

/// Moulded plastic of the corner protectors.
pub const CORNER_BODY: vg::Color = rgb(0x18, 0x18, 0x1A);
/// Sheen along the lit face of a corner protector.
pub const CORNER_LIT: vg::Color = rgb(0x4A, 0x4A, 0x50);
/// Outline where a corner protector meets the vinyl.
pub const CORNER_EDGE: vg::Color = rgb(0x0A, 0x0A, 0x0C);

/// The control panel's white face.
pub const PANEL_WHITE: vg::Color = rgb(0xF9, 0xF7, 0xF2);
/// The same face where it falls into shadow at the bottom of the panel.
pub const PANEL_SHADE: vg::Color = rgb(0xE4, 0xE1, 0xD9);
/// The heavy black outline around the control bar and between its cells.
pub const BAR_INK: vg::Color = rgb(0x10, 0x10, 0x12);
/// Screen-printed ink: glyphs and engraved detail.
pub const PANEL_INK: vg::Color = rgb(0x1A, 0x17, 0x14);
/// Unlit engraved index marks.
pub const PANEL_ENGRAVE: vg::Color = rgb(0x8C, 0x84, 0x74);

/// Main body of the moulded knob cap.
pub const KNOB_BODY: vg::Color = rgb(0x14, 0x14, 0x16);
/// Sheen where light catches the top-left of the cap.
pub const KNOB_HIGHLIGHT: vg::Color = rgb(0x5A, 0x5A, 0x60);
/// Outer rim line.
pub const KNOB_RIM: vg::Color = rgb(0x3C, 0x3C, 0x42);
/// Flute grooves in the rim.
pub const KNOB_FLUTE: vg::Color = rgb(0x2E, 0x2E, 0x34);
/// The white indicator line.
pub const KNOB_POINTER: vg::Color = rgb(0xF6, 0xF4, 0xEF);
/// Contact shadow beneath a knob.
pub const KNOB_SHADOW: vg::Color = rgb(0x40, 0x36, 0x26);

/// Bright core of the pilot jewel.
pub const AMBER: vg::Color = rgb(0xFF, 0xA8, 0x1E);
/// Deep amber at the jewel's edge and in its unlit state.
pub const AMBER_DEEP: vg::Color = rgb(0x8C, 0x3E, 0x00);
/// Chrome bezel around the jewel and the toggle hardware.
pub const CHROME_LIGHT: vg::Color = rgb(0xDA, 0xDD, 0xE2);
/// Shadowed side of the chrome.
pub const CHROME_DARK: vg::Color = rgb(0x6E, 0x74, 0x7E);
/// Black phenolic base a switch is mounted in.
pub const HARDWARE_BASE: vg::Color = rgb(0x1A, 0x1A, 0x1E);

/// Returns `colour` with its alpha scaled by `opacity`.
///
/// Vizia composites a view's own opacity, so every canvas-drawn paint has to
/// apply it manually or a faded container will leave its children fully opaque.
#[inline]
pub fn with_opacity(colour: vg::Color, opacity: f32) -> vg::Color {
    vg::Color {
        a: colour.a * opacity.clamp(0.0, 1.0),
        ..colour
    }
}

/// Linearly blends between two colours.
#[inline]
pub fn mix(from: vg::Color, to: vg::Color, t: f32) -> vg::Color {
    let t = t.clamp(0.0, 1.0);
    vg::Color {
        r: from.r + (to.r - from.r) * t,
        g: from.g + (to.g - from.g) * t,
        b: from.b + (to.b - from.b) * t,
        a: from.a + (to.a - from.a) * t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_maps_bytes_onto_the_unit_range() {
        let black = rgb(0, 0, 0);
        assert_eq!((black.r, black.g, black.b, black.a), (0.0, 0.0, 0.0, 1.0));
        let white = rgb(255, 255, 255);
        assert_eq!((white.r, white.g, white.b, white.a), (1.0, 1.0, 1.0, 1.0));
    }

    #[test]
    fn with_opacity_scales_alpha_only() {
        let faded = with_opacity(ORANGE, 0.5);
        assert!((faded.a - 0.5).abs() < 1.0e-6);
        assert_eq!(faded.r, ORANGE.r);
        assert_eq!(faded.g, ORANGE.g);
        assert_eq!(faded.b, ORANGE.b);
    }

    #[test]
    fn with_opacity_clamps_out_of_range_input() {
        assert_eq!(with_opacity(AMBER, 2.0).a, 1.0);
        assert_eq!(with_opacity(AMBER, -1.0).a, 0.0);
    }

    #[test]
    fn mix_interpolates_and_clamps() {
        let midpoint = mix(AMBER_DEEP, AMBER, 0.5);
        assert!((midpoint.r - (AMBER_DEEP.r + AMBER.r) * 0.5).abs() < 1.0e-6);
        let start = mix(AMBER_DEEP, AMBER, -3.0);
        assert_eq!(start.r, AMBER_DEEP.r);
        let end = mix(AMBER_DEEP, AMBER, 9.0);
        assert_eq!(end.r, AMBER.r);
    }

    /// Perceptual luminance, used to check contrast between panel elements.
    fn luminance(colour: vg::Color) -> f32 {
        0.2126 * colour.r + 0.7152 * colour.g + 0.0722 * colour.b
    }

    #[test]
    fn the_orange_family_is_actually_orange() {
        for (name, colour) in [
            ("band", ORANGE),
            ("deep", ORANGE_DEEP),
            ("amber", AMBER),
            ("tolex lit", TOLEX_LIT),
            ("tolex shade", TOLEX_SHADE),
            ("tolex grain", TOLEX_GRAIN),
            ("tolex edge", TOLEX_EDGE),
        ] {
            assert!(
                colour.r > colour.g && colour.g > colour.b,
                "{name} is not an orange hue"
            );
        }
    }

    #[test]
    fn the_vinyl_is_lit_from_above() {
        // The box is drawn with a top-to-bottom gradient, so the lit shade has
        // to be the brighter one or the light appears to come from the floor.
        assert!(
            luminance(TOLEX_LIT) > luminance(TOLEX_SHADE),
            "the tolex gradient is upside down"
        );
        // Both grain colours have to be visible against the vinyl without
        // turning it into a different material.
        let grain = luminance(TOLEX_SHADE) - luminance(TOLEX_GRAIN);
        assert!((0.03..0.35).contains(&grain), "grain contrast is {grain}");
        assert!(
            luminance(TOLEX_EDGE) < luminance(TOLEX_SHADE),
            "the edge shadow is lighter than the surface it shades"
        );
    }

    #[test]
    fn the_panel_is_a_warm_near_white() {
        for (name, colour) in [("panel", PANEL_WHITE), ("shade", PANEL_SHADE)] {
            assert!(luminance(colour) > 0.8, "{name} is too dark for the panel");
            assert!(colour.r >= colour.g, "{name} is not warm");
            assert!(colour.g >= colour.b, "{name} is not warm");
        }
        assert!(
            luminance(PANEL_WHITE) > luminance(PANEL_SHADE),
            "the panel gradient is upside down"
        );
        // The panel has to read as a lighter material than the vinyl it is
        // set into, which is the whole point of the aperture.
        assert!(luminance(PANEL_WHITE) > luminance(TOLEX_LIT) + 0.3);
    }

    #[test]
    fn the_corner_protectors_read_as_black_plastic() {
        assert!(luminance(CORNER_BODY) < 0.15, "the corners are not dark");
        assert!(
            luminance(CORNER_LIT) > luminance(CORNER_BODY),
            "the corner protectors have no highlight"
        );
        assert!(
            luminance(CORNER_EDGE) < luminance(CORNER_BODY),
            "the corner outline does not read against the moulding"
        );
    }

    #[test]
    fn every_marking_contrasts_with_the_surface_it_sits_on() {
        // Screen-printed ink and engraved marks against the white panel.
        for (name, ink) in [("ink", PANEL_INK), ("engrave", PANEL_ENGRAVE)] {
            let contrast = luminance(PANEL_WHITE) - luminance(ink);
            assert!(
                contrast > 0.15,
                "{name} is illegible on the panel: {contrast}"
            );
        }
        // Glyphs are printed on the orange cells too, so they have to survive
        // that background as well as the white one.
        let on_orange = luminance(BAND_ORANGE) - luminance(PANEL_INK);
        assert!(on_orange > 0.15, "ink is illegible on orange: {on_orange}");
        // The bar outline against both of the surfaces it separates.
        for (name, surface) in [("panel", PANEL_WHITE), ("band", BAND_ORANGE)] {
            let contrast = luminance(surface) - luminance(BAR_INK);
            assert!(contrast > 0.2, "the bar outline vanishes on {name}");
        }
        // The white pointer against the black knob cap.
        let pointer = luminance(KNOB_POINTER) - luminance(KNOB_BODY);
        assert!(pointer > 0.8, "pointer is invisible on the cap: {pointer}");
        // The knob cap against the orange cell it is mounted on.
        let cap = luminance(BAND_ORANGE) - luminance(KNOB_BODY);
        assert!(cap > 0.3, "knob cap does not stand out: {cap}");
    }

    #[test]
    fn chrome_has_a_light_and_a_dark_side() {
        let range = luminance(CHROME_LIGHT) - luminance(CHROME_DARK);
        assert!(range > 0.2, "chrome gradient is too flat: {range}");
        assert!(luminance(HARDWARE_BASE) < luminance(CHROME_DARK));
    }

    #[test]
    fn the_lit_jewel_is_brighter_than_the_unlit_one() {
        let range = luminance(AMBER) - luminance(AMBER_DEEP);
        assert!(range > 0.2, "the jewel barely changes when lit: {range}");
    }
}
