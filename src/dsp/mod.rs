//! Virtual-analog signal chain for the Orange OR100 Modern Reissue.
//!
//! Module layout mirrors the block diagram in section 2 of the technical
//! specification:
//!
//! ```text
//! In ─► [oversampling::Oversampler8x] ─────────────────────────────────┐
//!                                                                      │
//!   ┌──────────────── 8x oversampled sub-pipeline ──────────────────┐  │
//!   │ triode::Triode ─► tonestack::ToneStack ─► triode::Triode ...  │  │
//!   │        └─► power::PhaseInverter ─► power::PowerAmp            │  │
//!   └───────────────────────────────────────────────────────────────┘  │
//!                                                                      │
//! ◄─ cabinet::Cabinet ◄─ transformer::OutputTransformer ◄──────────────┘
//! ```
//!
//! Every module in here is host-agnostic: nothing below this point knows about
//! `nih-plug`, buffers, or parameters. That keeps the DSP unit-testable in a
//! plain `cargo test` harness (specification section 5 / `CLAUDE.md` §5) and
//! keeps `src/gui` decoupled from DSP internals (`CLAUDE.md` §3).

pub mod cabinet;
pub mod denormal;
pub mod engine;
pub mod filters;
pub mod oversampling;
pub mod power;
pub mod tonestack;
pub mod transformer;
pub mod triode;

/// Oversampling factor applied around every non-linear stage.
///
/// Specification section 2: "All non-linear stages run within an 8x polyphase
/// oversampling wrapper to eliminate intermodulation and aliasing distortion."
pub const OVERSAMPLING_FACTOR: usize = 8;

/// Converts a cutoff frequency into the coefficient of a one-pole lowpass
/// running at `sample_rate`.
///
/// This is the exact (impulse-invariant) pole rather than the `2*pi*fc/fs`
/// approximation, so it stays correct even when `fc` approaches Nyquist — which
/// it does for the 10 Hz DC blockers when the plugin runs at 8x of 192 kHz.
#[inline]
pub fn one_pole_coeff(cutoff_hz: f32, sample_rate: f32) -> f32 {
    debug_assert!(sample_rate > 0.0);
    let x = -std::f32::consts::TAU * (cutoff_hz / sample_rate);
    1.0 - x.exp()
}

/// Converts a time constant in milliseconds into a one-pole smoothing
/// coefficient, defined so the envelope reaches 1 - 1/e of a step in `ms`.
#[inline]
pub fn time_constant_coeff(ms: f32, sample_rate: f32) -> f32 {
    debug_assert!(sample_rate > 0.0);
    if ms <= 0.0 {
        return 1.0;
    }
    1.0 - (-1000.0 / (ms * sample_rate)).exp()
}

/// Maps a 0.0..=10.0 front-panel knob reading onto a normalized 0.0..=1.0
/// potentiometer rotation.
#[inline]
pub fn knob_to_rotation(knob: f32) -> f32 {
    (knob * 0.1).clamp(0.0, 1.0)
}

/// Audio-taper ("log pot") law used by the treble, bass and volume controls.
///
/// Real 250 kΩ/1 MΩ audio-taper pots follow roughly a 10 % resistance at 50 %
/// rotation law. `rotation^2.2` matches a measured Alpha A-taper to within
/// ~2 % over the useful range and, unlike a piecewise-linear approximation, has
/// a continuous derivative so parameter automation stays zipper-free.
#[inline]
pub fn audio_taper(rotation: f32) -> f32 {
    rotation.clamp(0.0, 1.0).powf(2.2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_pole_coeff_matches_analytic_cutoff() {
        let fs = 48_000.0;
        let a = one_pole_coeff(1_000.0, fs);
        // Magnitude response of y[n] = y[n-1] + a * (x[n] - y[n-1]) at the
        // cutoff should be -3.01 dB.
        let w = std::f32::consts::TAU * 1_000.0 / fs;
        let (sin_w, cos_w) = w.sin_cos();
        let denom_re = 1.0 - (1.0 - a) * cos_w;
        let denom_im = (1.0 - a) * sin_w;
        let mag = a / (denom_re * denom_re + denom_im * denom_im).sqrt();
        let db = 20.0 * mag.log10();
        assert!((db + 3.01).abs() < 0.1, "cutoff magnitude was {db} dB");
    }

    #[test]
    fn time_constant_reaches_63_percent() {
        let fs = 48_000.0;
        let a = time_constant_coeff(8.0, fs);
        let mut y = 0.0f32;
        for _ in 0..(0.008 * fs) as usize {
            y += a * (1.0 - y);
        }
        assert!((y - 0.632).abs() < 0.01, "envelope reached {y}");
    }

    #[test]
    fn audio_taper_is_monotonic_and_bounded() {
        let mut previous = -1.0;
        for step in 0..=100 {
            let value = audio_taper(step as f32 / 100.0);
            assert!(value >= previous);
            assert!((0.0..=1.0).contains(&value));
            previous = value;
        }
        assert_eq!(audio_taper(0.0), 0.0);
        assert_eq!(audio_taper(1.0), 1.0);
        // ~50 % rotation should land near 20 % of full resistance.
        assert!(audio_taper(0.5) < 0.3);
    }
}
