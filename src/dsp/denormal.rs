//! Denormal and NaN/Inf protection primitives.
//!
//! Recursive structures in this plugin (IIR tone stacks, DC blockers, the sag
//! envelope integrator, the coupling-capacitor state) all decay exponentially
//! towards zero. Once their state enters the subnormal range the x86 FPU falls
//! back to microcode and costs 100+ cycles per operation, which is fatal for the
//! CPU budget in section 5 of the technical specification.
//!
//! Two independent mitigations are applied, as required by `CLAUDE.md` §1:
//!
//! 1. [`DenormalGuard`] sets the SSE `FTZ` (flush-to-zero) and `DAZ`
//!    (denormals-are-zero) MXCSR bits for the duration of the audio callback and
//!    restores the host's original MXCSR on drop.
//! 2. [`flush`] and [`sanitize`] provide branch-cheap software fallbacks for
//!    targets without SSE and for hard NaN/Inf containment, which `FTZ`/`DAZ`
//!    do *not* cover.

/// Any magnitude below this is treated as silence by [`flush`].
///
/// `1e-20` sits far below the -300 dBFS noise floor of any real signal while
/// remaining several orders of magnitude above `f32::MIN_POSITIVE` (~1.18e-38),
/// so it clamps well before the subnormal range is reached.
pub const DENORMAL_THRESHOLD: f32 = 1.0e-20;

/// RAII guard that enables flush-to-zero and denormals-are-zero on x86/x86_64
/// and restores the previous MXCSR state when dropped.
///
/// Constructing and dropping this guard is allocation-free and lock-free, so it
/// is safe to create at the top of `Plugin::process`.
pub struct DenormalGuard {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    previous_mxcsr: u32,
}

/// Reads the SSE control/status register.
///
/// `std::arch::_mm_getcsr` is deprecated in favour of inline assembly, because
/// the intrinsic's semantics are not expressible to LLVM's optimiser. The
/// `stmxcsr`/`ldmxcsr` pair below is the replacement the deprecation notice
/// points at.
///
/// # Safety
///
/// Requires SSE, which is baseline on every target this `cfg` admits.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn read_mxcsr() -> u32 {
    let mut value: u32 = 0;
    // SAFETY: `stmxcsr` writes exactly four bytes to the supplied address, and
    // `value` is a live, correctly aligned `u32` on this stack frame.
    unsafe {
        std::arch::asm!(
            "stmxcsr [{ptr}]",
            ptr = in(reg) std::ptr::addr_of_mut!(value),
            options(nostack, preserves_flags),
        );
    }
    value
}

/// Writes the SSE control/status register.
///
/// # Safety
///
/// Requires SSE. The caller is responsible for restoring a sensible value;
/// [`DenormalGuard`] does so on drop.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn write_mxcsr(value: u32) {
    // SAFETY: `ldmxcsr` reads exactly four bytes from the supplied address, and
    // `value` is a live, correctly aligned `u32` on this stack frame.
    unsafe {
        std::arch::asm!(
            "ldmxcsr [{ptr}]",
            ptr = in(reg) std::ptr::addr_of!(value),
            options(nostack, preserves_flags),
        );
    }
}

impl DenormalGuard {
    /// Bit 15 of MXCSR: flush-to-zero for arithmetic *results*.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    const FTZ: u32 = 1 << 15;
    /// Bit 6 of MXCSR: denormals-are-zero for arithmetic *operands*.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    const DAZ: u32 = 1 << 6;

    /// Enables FTZ/DAZ for the lifetime of the returned guard.
    #[inline]
    pub fn new() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // SAFETY: these helpers only read and write the MXCSR control
            // register. SSE is guaranteed on x86_64 and is part of the baseline
            // for the `i686-*` targets this plugin supports. The original value
            // is restored in `Drop`.
            unsafe {
                let previous_mxcsr = read_mxcsr();
                write_mxcsr(previous_mxcsr | Self::FTZ | Self::DAZ);
                Self { previous_mxcsr }
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // AArch64 sets FZ via FPCR. Rust offers no stable intrinsic for it,
            // so on those targets the software fallbacks in `flush()` carry the
            // load; every recursive structure in `crate::dsp` calls it.
            Self {}
        }
    }
}

impl Default for DenormalGuard {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DenormalGuard {
    #[inline]
    fn drop(&mut self) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // SAFETY: restoring the exact value observed in `new()`.
            unsafe { write_mxcsr(self.previous_mxcsr) };
        }
    }
}

/// Branchlessly flushes subnormal magnitudes to exactly zero.
///
/// Used on the state of every recursive filter so that the plugin behaves
/// identically on targets where [`DenormalGuard`] is a no-op.
#[inline(always)]
pub fn flush(x: f32) -> f32 {
    // `f32::abs` compiles to an `andps`, and the comparison to a `cmpltss` +
    // `andnps`, so this is three branch-free instructions.
    if x.abs() < DENORMAL_THRESHOLD {
        0.0
    } else {
        x
    }
}

/// Replaces `NaN` and `±Inf` with `0.0` and hard-limits the magnitude.
///
/// `CLAUDE.md` §1 requires that no feedback or division path can emit a value
/// that would damage monitoring equipment. Every stage boundary in the amp
/// chain passes through this function, so a numerical blow-up anywhere is
/// contained to the sample it occurred on rather than latching into an IIR
/// state.
///
/// The `±32.0` bound is ~+30 dBFS: far above any musically meaningful level the
/// chain produces, so it never engages during normal operation.
///
/// This bound applies only to *normalized* audio. Nodes carrying real circuit
/// voltages — every triode plate, the phase-inverter legs, the tone stack —
/// swing hundreds of volts and must use [`sanitize_volts`] instead; clamping
/// them at 32 V would hard-limit the amplifier well below its own headroom.
#[inline(always)]
pub fn sanitize(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(-32.0, 32.0)
    } else {
        0.0
    }
}

/// Largest magnitude, in volts, any node inside the amplifier may reach.
///
/// The highest supply in the model is the 480 V power-stage rail; a preamp
/// plate hangs off 300 V and can only swing between ground and its supply.
/// 600 V therefore sits above every physically reachable node voltage while
/// still bounding a numerical blow-up to something finite.
pub const VOLTAGE_LIMIT: f32 = 600.0;

/// [`sanitize`] for signals expressed in real circuit volts.
///
/// Same `NaN`/`Inf` containment, but bounded by [`VOLTAGE_LIMIT`] rather than
/// by a normalized-audio ceiling, so a triode driven into cutoff can present
/// its full plate swing to the next stage instead of being brick-wall clipped.
#[inline(always)]
pub fn sanitize_volts(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(-VOLTAGE_LIMIT, VOLTAGE_LIMIT)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_kills_subnormals_but_keeps_audio() {
        assert_eq!(flush(1.0e-30), 0.0);
        assert_eq!(flush(-1.0e-30), 0.0);
        assert_eq!(flush(f32::MIN_POSITIVE / 2.0), 0.0);
        // -120 dBFS is still audible headroom and must survive untouched.
        assert_eq!(flush(1.0e-6), 1.0e-6);
        assert_eq!(flush(-0.5), -0.5);
    }

    #[test]
    fn sanitize_contains_non_finite_and_explosive_values() {
        assert_eq!(sanitize(f32::NAN), 0.0);
        assert_eq!(sanitize(f32::INFINITY), 0.0);
        assert_eq!(sanitize(f32::NEG_INFINITY), 0.0);
        assert_eq!(sanitize(1.0e9), 32.0);
        assert_eq!(sanitize(-1.0e9), -32.0);
        assert_eq!(sanitize(0.25), 0.25);
    }

    #[test]
    fn sanitize_volts_contains_blow_ups_without_clipping_plate_swings() {
        assert_eq!(sanitize_volts(f32::NAN), 0.0);
        assert_eq!(sanitize_volts(f32::INFINITY), 0.0);
        assert_eq!(sanitize_volts(f32::NEG_INFINITY), 0.0);
        assert_eq!(sanitize_volts(1.0e9), VOLTAGE_LIMIT);
        assert_eq!(sanitize_volts(-1.0e9), -VOLTAGE_LIMIT);
        // A 12AX7 plate cut off from a 300 V rail swings ~100 V above its
        // quiescent point; that must pass through untouched.
        assert_eq!(sanitize_volts(105.0), 105.0);
        assert_eq!(sanitize_volts(-150.0), -150.0);
    }

    #[test]
    fn guard_restores_previous_control_word() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // SAFETY: read-only access to MXCSR.
            let before = unsafe { read_mxcsr() };
            {
                let _guard = DenormalGuard::new();
                let during = unsafe { read_mxcsr() };
                assert_eq!(
                    during & (DenormalGuard::FTZ | DenormalGuard::DAZ),
                    DenormalGuard::FTZ | DenormalGuard::DAZ
                );
            }
            let after = unsafe { read_mxcsr() };
            assert_eq!(before, after);
        }
    }
}
