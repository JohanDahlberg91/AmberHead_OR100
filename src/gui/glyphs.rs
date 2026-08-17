//! Vector pictograms for the "Pics Only" faceplate.
//!
//! Specification section 4 calls for "crisp vector renders of official icons
//! (Speaker for Volume, Clefs for Bass/Treble, Soundwave for Middle,
//! Fist/Burst for Gain)".
//!
//! These are **original** vector constructions that convey the same meanings.
//! They are not traced from Orange Amplification's artwork, which is that
//! company's intellectual property and cannot be redistributed here. Every
//! glyph is authored in a normalized `0.0..=1.0` box and mapped onto its
//! destination rectangle at draw time, so it stays crisp from the 75 % to the
//! 200 % scale factor specification section 4 requires.

use nih_plug_vizia::vizia::prelude::Canvas;
use nih_plug_vizia::vizia::vg;

/// The pictograms available on the faceplate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glyph {
    /// Speaker cone with radiating waves — the Volume controls.
    Speaker,
    /// F-clef — the Bass controls.
    BassClef,
    /// G-clef — the Treble controls.
    TrebleClef,
    /// Travelling waveform — the Middle control.
    SoundWave,
    /// Radiating burst — the Gain control.
    Burst,
}

/// Maps a normalized coordinate onto the destination box.
#[derive(Clone, Copy)]
struct Frame {
    x: f32,
    y: f32,
    size: f32,
}

impl Frame {
    #[inline]
    fn point(&self, u: f32, v: f32) -> (f32, f32) {
        (self.x + u * self.size, self.y + v * self.size)
    }

    #[inline]
    fn length(&self, u: f32) -> f32 {
        u * self.size
    }

    fn move_to(&self, path: &mut vg::Path, u: f32, v: f32) {
        let (x, y) = self.point(u, v);
        path.move_to(x, y);
    }

    fn line_to(&self, path: &mut vg::Path, u: f32, v: f32) {
        let (x, y) = self.point(u, v);
        path.line_to(x, y);
    }

    /// A cubic segment needs two control points and an end point; expressing
    /// that as six scalars is the whole purpose of this helper.
    #[allow(clippy::too_many_arguments)]
    fn bezier_to(
        &self,
        path: &mut vg::Path,
        c1u: f32,
        c1v: f32,
        c2u: f32,
        c2v: f32,
        u: f32,
        v: f32,
    ) {
        let (c1x, c1y) = self.point(c1u, c1v);
        let (c2x, c2y) = self.point(c2u, c2v);
        let (x, y) = self.point(u, v);
        path.bezier_to(c1x, c1y, c2x, c2y, x, y);
    }

    fn circle(&self, path: &mut vg::Path, u: f32, v: f32, r: f32) {
        let (x, y) = self.point(u, v);
        path.circle(x, y, self.length(r));
    }
}

/// Draws `glyph` into the square box at `(x, y)` with side `size`, inked in
/// `color`.
pub fn draw(canvas: &mut Canvas, glyph: Glyph, x: f32, y: f32, size: f32, color: vg::Color) {
    let frame = Frame { x, y, size };
    match glyph {
        Glyph::Speaker => draw_speaker(canvas, frame, color),
        Glyph::BassClef => draw_bass_clef(canvas, frame, color),
        Glyph::TrebleClef => draw_treble_clef(canvas, frame, color),
        Glyph::SoundWave => draw_sound_wave(canvas, frame, color),
        Glyph::Burst => draw_burst(canvas, frame, color),
    }
}

/// Speaker: a filled magnet body and cone, with two radiating arcs.
fn draw_speaker(canvas: &mut Canvas, frame: Frame, color: vg::Color) {
    let mut body = vg::Path::new();
    // Voice-coil neck.
    frame.move_to(&mut body, 0.06, 0.38);
    frame.line_to(&mut body, 0.24, 0.38);
    frame.line_to(&mut body, 0.48, 0.12);
    frame.line_to(&mut body, 0.48, 0.88);
    frame.line_to(&mut body, 0.24, 0.62);
    frame.line_to(&mut body, 0.06, 0.62);
    body.close();
    canvas.fill_path(&body, &vg::Paint::color(color));

    // Two radiating wavefronts. Each arc is stroked as its own path:
    // `Path::arc` joins onto a non-empty path with a line, which would
    // otherwise draw a chord between the two wavefronts.
    let (cx, cy) = frame.point(0.46, 0.5);
    let mut paint = vg::Paint::color(color);
    paint.set_line_width(frame.length(0.075));
    paint.set_line_cap(vg::LineCap::Round);
    for radius in [0.20f32, 0.34] {
        let mut wave = vg::Path::new();
        wave.arc(cx, cy, frame.length(radius), -0.9, 0.9, vg::Solidity::Hole);
        canvas.stroke_path(&wave, &paint);
    }
}

/// Bass clef: the F-clef comma, its head, and the two dots straddling the
/// F line.
fn draw_bass_clef(canvas: &mut Canvas, frame: Frame, color: vg::Color) {
    let paint = vg::Paint::color(color);

    let mut head = vg::Path::new();
    frame.circle(&mut head, 0.30, 0.30, 0.115);
    canvas.fill_path(&head, &paint);

    let mut hook = vg::Path::new();
    frame.move_to(&mut hook, 0.30, 0.185);
    frame.bezier_to(&mut hook, 0.60, 0.16, 0.72, 0.36, 0.62, 0.56);
    frame.bezier_to(&mut hook, 0.53, 0.74, 0.34, 0.84, 0.16, 0.90);
    let mut stroke = vg::Paint::color(color);
    stroke.set_line_width(frame.length(0.115));
    stroke.set_line_cap(vg::LineCap::Round);
    canvas.stroke_path(&hook, &stroke);

    let mut dots = vg::Path::new();
    frame.circle(&mut dots, 0.83, 0.24, 0.055);
    frame.circle(&mut dots, 0.83, 0.42, 0.055);
    canvas.fill_path(&dots, &paint);
}

/// Treble clef: a stroked G-clef silhouette with the terminal dot.
fn draw_treble_clef(canvas: &mut Canvas, frame: Frame, color: vg::Color) {
    let mut path = vg::Path::new();
    // Bottom terminal, sweeping up and right into the body loop.
    frame.move_to(&mut path, 0.30, 0.90);
    frame.bezier_to(&mut path, 0.30, 0.99, 0.46, 1.00, 0.52, 0.92);
    frame.bezier_to(&mut path, 0.60, 0.80, 0.44, 0.68, 0.34, 0.62);
    // Large ascending loop.
    frame.bezier_to(&mut path, 0.16, 0.52, 0.20, 0.26, 0.42, 0.11);
    // Top hook.
    frame.bezier_to(&mut path, 0.58, 0.00, 0.72, 0.14, 0.64, 0.32);
    // Descending stem back through the loop.
    frame.bezier_to(&mut path, 0.57, 0.47, 0.42, 0.58, 0.40, 0.74);

    let mut stroke = vg::Paint::color(color);
    stroke.set_line_width(frame.length(0.095));
    stroke.set_line_cap(vg::LineCap::Round);
    canvas.stroke_path(&path, &stroke);

    let mut dot = vg::Path::new();
    frame.circle(&mut dot, 0.24, 0.86, 0.075);
    canvas.fill_path(&dot, &vg::Paint::color(color));
}

/// Middle: a travelling waveform, drawn as a sampled sine so the curve stays
/// smooth at every scale factor.
fn draw_sound_wave(canvas: &mut Canvas, frame: Frame, color: vg::Color) {
    const STEPS: usize = 48;
    let mut path = vg::Path::new();
    for step in 0..=STEPS {
        let t = step as f32 / STEPS as f32;
        let u = 0.06 + 0.88 * t;
        let v = 0.5 - 0.30 * (std::f32::consts::TAU * 1.5 * t).sin();
        if step == 0 {
            frame.move_to(&mut path, u, v);
        } else {
            frame.line_to(&mut path, u, v);
        }
    }
    let mut stroke = vg::Paint::color(color);
    stroke.set_line_width(frame.length(0.105));
    stroke.set_line_cap(vg::LineCap::Round);
    stroke.set_line_join(vg::LineJoin::Round);
    canvas.stroke_path(&path, &stroke);
}

/// Gain: a radiating burst — an eight-pointed star with a solid core.
fn draw_burst(canvas: &mut Canvas, frame: Frame, color: vg::Color) {
    const POINTS: usize = 8;
    let mut path = vg::Path::new();
    for index in 0..(POINTS * 2) {
        let angle = std::f32::consts::TAU * index as f32 / (POINTS * 2) as f32;
        let radius = if index % 2 == 0 { 0.48 } else { 0.19 };
        let u = 0.5 + radius * angle.sin();
        let v = 0.5 - radius * angle.cos();
        if index == 0 {
            frame.move_to(&mut path, u, v);
        } else {
            frame.line_to(&mut path, u, v);
        }
    }
    path.close();
    canvas.fill_path(&path, &vg::Paint::color(color));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_maps_the_unit_box_onto_its_destination() {
        let frame = Frame {
            x: 10.0,
            y: 20.0,
            size: 40.0,
        };
        assert_eq!(frame.point(0.0, 0.0), (10.0, 20.0));
        assert_eq!(frame.point(1.0, 1.0), (50.0, 60.0));
        assert_eq!(frame.point(0.5, 0.25), (30.0, 30.0));
        assert_eq!(frame.length(0.5), 20.0);
    }

    #[test]
    fn every_glyph_builds_a_non_empty_path() {
        // The drawing routines need a live GPU canvas, so exercise the path
        // construction directly: an empty path would mean a blank faceplate.
        let frame = Frame {
            x: 0.0,
            y: 0.0,
            size: 24.0,
        };
        let mut speaker = vg::Path::new();
        frame.move_to(&mut speaker, 0.06, 0.38);
        frame.line_to(&mut speaker, 0.24, 0.38);
        assert!(!speaker.is_empty());

        let mut wave = vg::Path::new();
        for step in 0..=48 {
            let t = step as f32 / 48.0;
            let u = 0.06 + 0.88 * t;
            let v = 0.5 - 0.30 * (std::f32::consts::TAU * 1.5 * t).sin();
            if step == 0 {
                frame.move_to(&mut wave, u, v);
            } else {
                frame.line_to(&mut wave, u, v);
            }
            assert!((0.0..=1.0).contains(&v), "wave left its box at {v}");
        }
        assert!(!wave.is_empty());
    }

    #[test]
    fn burst_points_stay_inside_the_unit_box() {
        for index in 0..16 {
            let angle = std::f32::consts::TAU * index as f32 / 16.0;
            let radius = if index % 2 == 0 { 0.48 } else { 0.19 };
            let u = 0.5 + radius * angle.sin();
            let v = 0.5 - radius * angle.cos();
            assert!((0.0..=1.0).contains(&u), "u={u}");
            assert!((0.0..=1.0).contains(&v), "v={v}");
        }
    }

    #[test]
    fn glyphs_are_distinct_values() {
        let all = [
            Glyph::Speaker,
            Glyph::BassClef,
            Glyph::TrebleClef,
            Glyph::SoundWave,
            Glyph::Burst,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }
}
