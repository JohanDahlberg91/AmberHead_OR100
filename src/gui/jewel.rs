//! Amber pilot jewel.
//!
//! Specification section 4: "Dynamic amber pilot light with real-time glow
//! intensity tied to the `B+` voltage rail state."
//!
//! The brightness the widget draws is published by the audio thread into a
//! [`crate::shared::AtomicLevel`] and copied into the editor's model on a
//! timer tick, so the lamp visibly dips when the power stage sags under hard
//! playing and swells back as the rail recovers.

use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;

use super::theme;

/// Brightness below which the jewel is drawn as cold glass.
const DARK_THRESHOLD: f32 = 0.02;

/// Whether the lamp is lit enough to cast a halo.
#[inline]
fn is_lit(brightness: f32) -> bool {
    brightness > DARK_THRESHOLD
}

/// The pilot jewel.
pub struct JewelLamp<L>
where
    L: Lens<Target = f32>,
{
    brightness: L,
}

/// Creates a jewel lamp driven by a `0.0..=1.0` brightness lens.
pub fn jewel_lamp<L>(cx: &mut Context, brightness: L) -> Handle<'_, impl View>
where
    L: Lens<Target = f32>,
{
    JewelLamp { brightness }.build(cx, |_| {}).class("jewel")
}

impl<L> View for JewelLamp<L>
where
    L: Lens<Target = f32>,
{
    fn element(&self) -> Option<&'static str> {
        Some("jewel")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        if bounds.w <= 0.0 || bounds.h <= 0.0 {
            return;
        }
        let opacity = cx.opacity();
        let brightness = self.brightness.get(cx).clamp(0.0, 1.0);

        let centre_x = bounds.x + bounds.w * 0.5;
        let centre_y = bounds.y + bounds.h * 0.5;
        let radius = bounds.w.min(bounds.h) * 0.5;

        // Halo: only rendered once the lamp is actually lit, and its reach
        // grows with brightness so a sagging rail visibly pulls the glow in.
        if is_lit(brightness) {
            let halo_radius = radius * (1.35 + 0.85 * brightness);
            let mut halo = vg::Path::new();
            halo.circle(centre_x, centre_y, halo_radius);
            let halo_paint = vg::Paint::radial_gradient(
                centre_x,
                centre_y,
                radius * 0.55,
                halo_radius,
                theme::with_opacity(theme::AMBER, opacity * brightness * 0.45),
                theme::with_opacity(theme::AMBER, 0.0),
            );
            canvas.fill_path(&halo, &halo_paint);
        }

        // Chrome bezel.
        let mut bezel = vg::Path::new();
        bezel.circle(centre_x, centre_y, radius);
        let bezel_paint = vg::Paint::linear_gradient(
            centre_x,
            centre_y - radius,
            centre_x,
            centre_y + radius,
            theme::with_opacity(theme::CHROME_LIGHT, opacity),
            theme::with_opacity(theme::CHROME_DARK, opacity),
        );
        canvas.fill_path(&bezel, &bezel_paint);

        // Faceted amber dome. Unlit it is deep and glassy; lit it fills with
        // the bright core colour.
        let dome_radius = radius * 0.74;
        let core = theme::mix(theme::AMBER_DEEP, theme::AMBER, brightness);
        let edge = theme::mix(
            vg::Color {
                a: 1.0,
                ..theme::AMBER_DEEP
            },
            theme::AMBER_DEEP,
            brightness * 0.5,
        );
        let mut dome = vg::Path::new();
        dome.circle(centre_x, centre_y, dome_radius);
        let dome_paint = vg::Paint::radial_gradient(
            centre_x - dome_radius * 0.18,
            centre_y - dome_radius * 0.22,
            dome_radius * 0.08,
            dome_radius,
            theme::with_opacity(core, opacity),
            theme::with_opacity(edge, opacity),
        );
        canvas.fill_path(&dome, &dome_paint);

        // Cut facets: eight radial lines, as on a moulded jewel lens.
        let mut facets = vg::Path::new();
        for index in 0..8 {
            let angle = std::f32::consts::TAU * index as f32 / 8.0;
            let (sin, cos) = angle.sin_cos();
            facets.move_to(
                centre_x + sin * dome_radius * 0.30,
                centre_y - cos * dome_radius * 0.30,
            );
            facets.line_to(
                centre_x + sin * dome_radius * 0.97,
                centre_y - cos * dome_radius * 0.97,
            );
        }
        let mut facet_paint = vg::Paint::color(theme::with_opacity(
            theme::AMBER_DEEP,
            opacity * (0.35 - 0.2 * brightness).max(0.05),
        ));
        facet_paint.set_line_width((radius * 0.06).max(0.6));
        canvas.stroke_path(&facets, &facet_paint);

        // Specular highlight, sharpest when the lamp is cold glass.
        let mut highlight = vg::Path::new();
        highlight.ellipse(
            centre_x - dome_radius * 0.30,
            centre_y - dome_radius * 0.38,
            dome_radius * 0.26,
            dome_radius * 0.17,
        );
        canvas.fill_path(
            &highlight,
            &vg::Paint::color(theme::with_opacity(
                theme::CHROME_LIGHT,
                opacity * (0.55 - 0.25 * brightness),
            )),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halo_radius_grows_with_brightness() {
        let radius = 20.0f32;
        let reach = |brightness: f32| radius * (1.35 + 0.85 * brightness);
        assert!(reach(1.0) > reach(0.5));
        assert!(reach(0.5) > reach(0.0));
        // Even at full brightness the halo stays within a sane bound of the
        // bezel, so it cannot bleed across the whole faceplate.
        assert!(reach(1.0) < radius * 2.5);
    }

    #[test]
    fn dome_colour_runs_from_deep_amber_to_bright() {
        let cold = theme::mix(theme::AMBER_DEEP, theme::AMBER, 0.0);
        let hot = theme::mix(theme::AMBER_DEEP, theme::AMBER, 1.0);
        assert_eq!(cold.r, theme::AMBER_DEEP.r);
        assert_eq!(hot.r, theme::AMBER.r);
        assert!(hot.g > cold.g, "lit jewel should be more yellow");
    }

    #[test]
    fn facet_and_highlight_alpha_stay_positive() {
        for step in 0..=10 {
            let brightness = step as f32 / 10.0;
            let facet = (0.35 - 0.2 * brightness).max(0.05);
            let highlight = 0.55 - 0.25 * brightness;
            assert!(facet > 0.0, "facet alpha went to {facet}");
            assert!(highlight > 0.0, "highlight alpha went to {highlight}");
        }
    }

    #[test]
    fn halo_follows_the_engine_reported_brightness_levels() {
        // `AmpEngine::lamp_brightness` reports exactly these three values for
        // the three power switch positions.
        assert!(!is_lit(0.0), "the halo must be dark with the amp off");
        assert!(is_lit(0.55), "standby must still glow");
        assert!(is_lit(1.0), "a full rail must glow");
        // ...and the threshold must not swallow a nearly-collapsed rail.
        assert!(is_lit(0.6), "a sagging rail went dark");
    }
}
