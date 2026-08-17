//! Faceplate palette.
//!
//! Specification section 4: "Textured cream/ivory enamel panel with top and
//! bottom Orange-stripe framing", "fluted black pointer knobs with white
//! indicator lines", "Dynamic amber pilot light".
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

/// Orange stripe framing the top and bottom of the chassis.
pub const ORANGE: vg::Color = rgb(0xE8, 0x62, 0x0E);
/// Deeper orange used for the stripe's inner shadow line.
pub const ORANGE_DEEP: vg::Color = rgb(0xB4, 0x46, 0x06);
/// Cream enamel of the main panel.
pub const IVORY: vg::Color = rgb(0xF2, 0xE9, 0xD6);
/// Slightly darker cream used for the panel's texture speckle.
pub const IVORY_SHADE: vg::Color = rgb(0xE3, 0xD8, 0xC0);
/// Screen-printed ink: glyphs and engraved detail.
pub const PANEL_INK: vg::Color = rgb(0x2A, 0x24, 0x1C);
/// Unlit engraved index marks.
pub const PANEL_ENGRAVE: vg::Color = rgb(0xB0, 0xA4, 0x8C);

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
        for (name, colour) in [("stripe", ORANGE), ("deep", ORANGE_DEEP), ("amber", AMBER)] {
            assert!(
                colour.r > colour.g && colour.g > colour.b,
                "{name} is not an orange hue"
            );
        }
    }

    #[test]
    fn the_enamel_is_a_warm_near_white() {
        for (name, colour) in [("ivory", IVORY), ("shade", IVORY_SHADE)] {
            assert!(luminance(colour) > 0.8, "{name} is too dark for enamel");
            assert!(colour.r >= colour.g, "{name} is not warm");
            assert!(colour.g >= colour.b, "{name} is not warm");
        }
    }

    #[test]
    fn every_marking_contrasts_with_the_surface_it_sits_on() {
        // Screen-printed ink and engraved marks against the enamel panel.
        for (name, ink) in [("ink", PANEL_INK), ("engrave", PANEL_ENGRAVE)] {
            let contrast = luminance(IVORY) - luminance(ink);
            assert!(
                contrast > 0.15,
                "{name} is illegible on the panel: {contrast}"
            );
        }
        // The white pointer against the black knob cap.
        let pointer = luminance(KNOB_POINTER) - luminance(KNOB_BODY);
        assert!(pointer > 0.8, "pointer is invisible on the cap: {pointer}");
        // The knob cap against the panel it is mounted on.
        let cap = luminance(IVORY) - luminance(KNOB_BODY);
        assert!(cap > 0.7, "knob cap does not stand out: {cap}");
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
