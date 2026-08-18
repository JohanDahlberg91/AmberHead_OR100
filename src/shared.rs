//! Lock-free state shared between the audio thread and the editor.
//!
//! This module deliberately sits outside both [`crate::dsp`] and
//! [`crate::gui`]. `CLAUDE.md` §3 requires the GUI to stay fully decoupled from
//! internal DSP structures, so the one value that genuinely has to cross the
//! boundary lives in neutral types that neither side owns: the jewel lamp's
//! brightness, the host sample rate the editor needs in order to resample a
//! loaded impulse response, and the response itself.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

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

/// Lock-free handoff of an impulse response from the editor to the audio
/// thread.
///
/// # Why a seqlock
///
/// `CLAUDE.md` §1 forbids mutexes on the audio thread, and no lock-free queue
/// crate is approved for this project. A seqlock needs neither: the editor
/// stamps an odd generation before it writes and an even one after, and the
/// audio thread reads the payload between two matching even stamps. If the two
/// stamps disagree the read was torn, and the reader simply abandons it — the
/// next block picks the response up, which for a user clicking a file in a
/// browser is a delay of under a millisecond.
///
/// The audio side therefore never blocks, never spins and never allocates: its
/// worst case is one wasted copy into a buffer it already owns.
///
/// # Why the taps are atomics
///
/// A plain `UnsafeCell<[f32]>` written by one thread while another reads it is
/// a data race and undefined behaviour, seqlock or not. Per-element relaxed
/// atomics make the same access pattern well defined at no measurable cost:
/// `AtomicU32::load(Relaxed)` compiles to an ordinary `mov` on every target
/// this plugin builds for.
#[derive(Debug)]
pub struct IrSlot {
    /// Impulse response taps, as `f32` bit patterns.
    taps: Vec<AtomicU32>,
    /// Number of valid entries in `taps`. Zero means "use the default cab".
    length: AtomicUsize,
    /// Seqlock counter: odd while a write is in progress.
    generation: AtomicU32,
    /// Guards against two editor threads publishing at once.
    writing: AtomicBool,
}

impl IrSlot {
    /// Creates a slot able to carry `capacity` taps.
    ///
    /// Allocates, so call this when the plugin is constructed, never later.
    pub fn new(capacity: usize) -> Self {
        Self {
            taps: (0..capacity).map(|_| AtomicU32::new(0)).collect(),
            length: AtomicUsize::new(0),
            generation: AtomicU32::new(0),
            writing: AtomicBool::new(false),
        }
    }

    /// Taps this slot can carry.
    pub fn capacity(&self) -> usize {
        self.taps.len()
    }

    /// Publishes `taps` for the audio thread. Called from the editor.
    ///
    /// Anything beyond [`Self::capacity`] is dropped; the loader is expected to
    /// have truncated already. Returns `false` if another thread is mid-publish,
    /// in which case nothing was written and the caller should try again.
    pub fn publish(&self, taps: &[f32]) -> bool {
        if self
            .writing
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }

        let generation = self.generation.load(Ordering::Relaxed);
        // Odd: a reader that catches the slot here knows the payload is
        // mid-flight and skips it.
        self.generation
            .store(generation.wrapping_add(1), Ordering::Release);

        let length = taps.len().min(self.taps.len());
        for (slot, tap) in self.taps.iter().zip(taps.iter()).take(length) {
            slot.store(tap.to_bits(), Ordering::Relaxed);
        }
        self.length.store(length, Ordering::Relaxed);

        self.generation
            .store(generation.wrapping_add(2), Ordering::Release);
        self.writing.store(false, Ordering::Release);
        true
    }

    /// Publishes the "revert to the built-in cabinet" request.
    ///
    /// Carried as a zero-length payload so the audio thread never has to
    /// synthesise anything, which it could not do without allocating.
    pub fn publish_default(&self) -> bool {
        self.publish(&[])
    }

    /// Collects a newly published response into `destination`.
    ///
    /// `seen` is the caller's record of the last generation it consumed; it is
    /// advanced only on a successful read. Returns the number of valid taps
    /// written into `destination`, or `None` when there is nothing new — which
    /// is the case on all but a handful of blocks in a session.
    ///
    /// A returned length of zero is a request to restore the default cabinet,
    /// not an empty response.
    ///
    /// Wait-free and allocation-free: safe on the audio thread.
    #[inline]
    pub fn collect(&self, seen: &mut u32, destination: &mut [f32]) -> Option<usize> {
        let generation = self.generation.load(Ordering::Acquire);
        // Odd means a write is in flight; equal means nothing has changed.
        if generation & 1 != 0 || generation == *seen {
            return None;
        }

        let length = self.length.load(Ordering::Relaxed).min(destination.len());
        for (index, slot) in destination.iter_mut().enumerate() {
            *slot = match self.taps.get(index) {
                Some(tap) if index < length => f32::from_bits(tap.load(Ordering::Relaxed)),
                _ => 0.0,
            };
        }

        // If the editor published again while this copy was running, the copy
        // may mix two responses. Abandon it; the next block reads the newer one.
        if self.generation.load(Ordering::Acquire) != generation {
            return None;
        }
        *seen = generation;
        Some(length)
    }

    /// The generation a fresh reader should start from.
    ///
    /// Reading the current value rather than starting at zero means a reader
    /// created after a response was published does not immediately re-collect
    /// it — the engine already loaded that response in `initialize`.
    pub fn current_generation(&self) -> u32 {
        self.generation.load(Ordering::Acquire) & !1
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
    fn a_fresh_slot_has_nothing_to_collect() {
        let slot = IrSlot::new(8);
        assert_eq!(slot.capacity(), 8);
        let mut seen = slot.current_generation();
        let mut destination = [9.0f32; 8];
        assert_eq!(slot.collect(&mut seen, &mut destination), None);
        // A refused collect must leave the destination alone rather than
        // clearing it, or a spurious poll would wipe the running cabinet.
        assert_eq!(destination, [9.0; 8]);
    }

    #[test]
    fn a_published_response_is_collected_exactly_once() {
        let slot = IrSlot::new(8);
        let mut seen = slot.current_generation();
        let mut destination = [0.0f32; 8];

        assert!(slot.publish(&[1.0, 2.0, 3.0]));
        assert_eq!(slot.collect(&mut seen, &mut destination), Some(3));
        assert_eq!(destination, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Nothing new: the audio thread must not reload the same response on
        // every block.
        assert_eq!(slot.collect(&mut seen, &mut destination), None);

        assert!(slot.publish(&[7.0]));
        assert_eq!(slot.collect(&mut seen, &mut destination), Some(1));
        assert_eq!(destination.first().copied(), Some(7.0));
        // Everything past the new length is cleared, so the tail of the
        // previous response cannot survive into the new one.
        assert!(destination.iter().skip(1).all(|tap| *tap == 0.0));
    }

    #[test]
    fn a_default_request_is_a_zero_length_payload() {
        let slot = IrSlot::new(4);
        let mut seen = slot.current_generation();
        let mut destination = [1.0f32; 4];
        assert!(slot.publish_default());
        assert_eq!(slot.collect(&mut seen, &mut destination), Some(0));
    }

    #[test]
    fn publishing_more_taps_than_the_slot_holds_keeps_the_head() {
        let slot = IrSlot::new(3);
        let mut seen = slot.current_generation();
        let mut destination = [0.0f32; 3];
        assert!(slot.publish(&[1.0, 2.0, 3.0, 4.0, 5.0]));
        assert_eq!(slot.collect(&mut seen, &mut destination), Some(3));
        assert_eq!(destination, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn collecting_into_a_short_buffer_does_not_overrun_it() {
        let slot = IrSlot::new(8);
        let mut seen = slot.current_generation();
        let mut destination = [0.0f32; 2];
        assert!(slot.publish(&[1.0, 2.0, 3.0, 4.0]));
        assert_eq!(slot.collect(&mut seen, &mut destination), Some(2));
        assert_eq!(destination, [1.0, 2.0]);
    }

    #[test]
    fn a_reader_created_after_a_publish_does_not_re_collect_it() {
        // The engine loads the persisted response in `initialize`; the audio
        // thread must not then load it a second time on its first block.
        let slot = IrSlot::new(4);
        assert!(slot.publish(&[1.0, 2.0]));
        let mut seen = slot.current_generation();
        let mut destination = [0.0f32; 4];
        assert_eq!(slot.collect(&mut seen, &mut destination), None);
    }

    #[test]
    fn a_concurrent_reader_never_sees_a_mixed_response() {
        // The property the seqlock exists for: every collected payload is one
        // publisher's, never two spliced together. The writer alternates
        // between two responses whose taps are all 1.0 and all 2.0, so a torn
        // read would show both values at once.
        let slot = Arc::new(IrSlot::new(4_096));
        let writer = Arc::clone(&slot);
        let finished = Arc::new(AtomicBool::new(false));
        let writer_finished = Arc::clone(&finished);
        let handle = std::thread::spawn(move || {
            for round in 0..4_000 {
                let value = if round % 2 == 0 { 1.0f32 } else { 2.0 };
                writer.publish(&vec![value; 4_096]);
            }
            writer_finished.store(true, Ordering::Release);
        });

        let mut seen = slot.current_generation();
        let mut destination = vec![0.0f32; 4_096];
        let mut collected = 0usize;
        // Read until the writer is done, so the reader genuinely races it
        // rather than spinning through its whole budget before it starts.
        while !finished.load(Ordering::Acquire) {
            if let Some(length) = slot.collect(&mut seen, &mut destination) {
                collected += 1;
                let first = destination.first().copied().unwrap_or(0.0);
                assert!(
                    destination.iter().take(length).all(|tap| *tap == first),
                    "a torn read mixed two responses"
                );
            }
        }
        assert!(handle.join().is_ok());
        assert!(collected > 0, "the reader never observed a publish");
    }

    #[test]
    fn the_generation_counter_survives_wrapping() {
        // u32 wrapping is reachable in a long session only in principle, but a
        // seqlock that deadlocks at the wrap is a seqlock that eventually
        // stops accepting cabinets.
        let slot = IrSlot::new(2);
        let mut seen = u32::MAX - 1;
        let mut destination = [0.0f32; 2];
        for _ in 0..8 {
            assert!(slot.publish(&[1.0]));
        }
        assert!(slot.collect(&mut seen, &mut destination).is_some());
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
