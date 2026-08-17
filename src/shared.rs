//! Lock-free state shared between the audio thread and the editor.
//!
//! This module deliberately sits outside both [`crate::dsp`] and
//! [`crate::gui`]. `CLAUDE.md` §3 requires the GUI to stay fully decoupled from
//! internal DSP structures, so the one value that genuinely has to cross the
//! boundary — the jewel lamp's brightness — lives in a neutral type that
//! neither side owns.

use std::sync::atomic::{AtomicU32, Ordering};

/// Wait-free `f32` cell.
///
/// `CLAUDE.md` §1 forbids mutexes between the audio and UI threads. An
/// `AtomicU32` holding the float's bit pattern is wait-free on every supported
/// target and needs no additional dependency. `Relaxed` ordering is correct
/// here: the value is a standalone display quantity that guards no other
/// memory, and a reader that observes a slightly stale brightness is
/// indistinguishable from one that redrew a frame earlier.
#[derive(Debug)]
pub struct AtomicLevel(AtomicU32);

impl AtomicLevel {
    /// Creates a cell holding `value`.
    pub fn new(value: f32) -> Self {
        Self(AtomicU32::new(value.to_bits()))
    }

    /// Publishes `value`. Called from the audio thread.
    #[inline]
    pub fn store(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Reads the most recently published value. Called from the UI thread.
    #[inline]
    pub fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
}

impl Default for AtomicLevel {
    fn default() -> Self {
        Self::new(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn round_trips_values_including_negatives_and_zero() {
        let cell = AtomicLevel::new(0.25);
        assert_eq!(cell.load(), 0.25);
        cell.store(-3.5);
        assert_eq!(cell.load(), -3.5);
        cell.store(0.0);
        assert_eq!(cell.load(), 0.0);
        assert_eq!(AtomicLevel::default().load(), 0.0);
    }

    #[test]
    fn is_shareable_across_threads() {
        let cell = Arc::new(AtomicLevel::new(0.0));
        let writer = Arc::clone(&cell);
        let handle = std::thread::spawn(move || {
            for step in 0..1_000 {
                writer.store(step as f32 / 1_000.0);
            }
        });
        for _ in 0..1_000 {
            let observed = cell.load();
            assert!((0.0..=1.0).contains(&observed));
        }
        assert!(handle.join().is_ok());
    }
}
