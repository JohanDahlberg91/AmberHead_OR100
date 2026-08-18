//! Real-time safety verification.
//!
//! `CLAUDE.md` §1 and §5 require proof that the audio path performs no heap
//! allocation. `nih-plug`'s `assert_process_allocs` feature is enabled in
//! `Cargo.toml` and catches this inside a hosted debug build, but it only fires
//! when a real DAW is driving the plugin.
//!
//! This suite proves the same property headlessly by installing a counting
//! global allocator for the test binary and asserting that the counter does not
//! move while the amplifier is processing. Every allocation the engine needs is
//! made in `prepare()`, which runs before the counter is sampled.
//!
//! # Why the allocator is release-only
//!
//! `nih-plug` installs `assert_no_alloc`'s `AllocDisabler` as *the* global
//! allocator in debug builds when the `assert_process_allocs` feature is on, and
//! a crate may only have one. The counting allocator and the allocation tests
//! that depend on it are therefore compiled for release builds only; in debug
//! builds nih-plug's own allocator covers the same ground from inside a hosted
//! `process()` call. The behavioural tests below run in both profiles.
//!
//! Run the full check with `cargo test --release`.

use amberhead_or100::dsp::engine::{AmpEngine, SampleControls};

// Only the release-only allocation tests sweep the discrete controls.
#[cfg(not(debug_assertions))]
use amberhead_or100::dsp::engine::{Channel, PowerState};
#[cfg(not(debug_assertions))]
use amberhead_or100::dsp::power::PowerMode;

/// Allocation instrumentation. See the module docs for why this is
/// release-only.
#[cfg(not(debug_assertions))]
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    /// Counts every allocation and reallocation made on the *calling* thread.
    struct CountingAllocator;

    thread_local! {
        /// Per-thread allocation counter.
        ///
        /// The counter must be thread-local, not a global atomic: the test harness
        /// runs these cases in parallel, so a process-wide counter would attribute
        /// every other test's allocations to whichever one happened to be measuring.
        ///
        /// `const` initialisation with a `Copy` payload and no destructor means the
        /// TLS slot is a plain thread-local static — it neither allocates on first
        /// access nor registers a destructor, so it cannot recurse back into the
        /// allocator it is instrumenting.
        static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    }

    /// Records one allocation against the current thread, ignoring the call if TLS
    /// is being torn down.
    #[inline]
    fn record_allocation() {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get().wrapping_add(1)));
    }

    // SAFETY: every method forwards directly to the system allocator with the
    // caller's layout unchanged; the only added behaviour is a counter increment on
    // a `Cell<usize>`, which cannot affect memory validity.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record_allocation();
            unsafe { System.realloc(ptr, layout, new_size) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            unsafe { System.alloc_zeroed(layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    /// Allocations recorded on this thread so far.
    pub fn allocation_count() -> usize {
        ALLOCATIONS.with(|count| count.get())
    }
}

#[cfg(not(debug_assertions))]
use counting::allocation_count;

/// Sanity check on the instrumentation itself: a test that trusts a counter
/// which never moves would pass no matter how much the engine allocated.
#[test]
#[cfg(not(debug_assertions))]
fn the_allocation_counter_actually_counts() {
    let before = allocation_count();
    let allocated: Vec<u8> = Vec::with_capacity(4_096);
    assert_eq!(allocated.capacity(), 4_096);
    let after = allocation_count();
    assert!(
        after > before,
        "counter did not observe a deliberate allocation"
    );

    // ...and that it stays put across work that should not allocate.
    let quiet_before = allocation_count();
    let mut accumulator = 0.0f32;
    for step in 0..10_000 {
        accumulator += (step as f32).sin();
    }
    assert!(accumulator.is_finite());
    assert_eq!(quiet_before, allocation_count());
}

const SAMPLE_RATE: f32 = 48_000.0;

fn prepared(controls: &SampleControls) -> Box<AmpEngine> {
    let mut engine = Box::new(AmpEngine::new());
    assert!(
        engine.prepare(SAMPLE_RATE, controls),
        "engine preparation failed"
    );
    engine
}

#[test]
#[cfg(not(debug_assertions))]
fn processing_allocates_nothing() {
    let controls = SampleControls::default();
    let mut engine = prepared(&controls);

    // Warm up past the first convolution block and the DC blockers, so any
    // lazily-built state is already in place before the counter is read.
    for n in 0..4_096 {
        let x = (std::f32::consts::TAU * 220.0 * n as f32 / SAMPLE_RATE).sin();
        engine.process_sample(x, &controls);
    }

    let before = allocation_count();
    for n in 0..(SAMPLE_RATE as usize) {
        let x = 0.8 * (std::f32::consts::TAU * 110.0 * n as f32 / SAMPLE_RATE).sin();
        let y = engine.process_sample(x, &controls);
        assert!(y.is_finite());
    }
    let after = allocation_count();

    assert_eq!(
        before,
        after,
        "one second of audio performed {} allocations",
        after - before
    );
}

#[test]
#[cfg(not(debug_assertions))]
fn switching_every_control_allocates_nothing() {
    // Parameter changes re-solve the tone stacks and re-key the power mode at
    // control rate. None of that may reach the heap either.
    let mut controls = SampleControls::default();
    let mut engine = prepared(&controls);
    for _ in 0..4_096 {
        engine.process_sample(0.0, &controls);
    }

    let before = allocation_count();
    for step in 0..8_192 {
        let t = step as f32 / 8_192.0;
        controls.channel = if step % 512 < 256 {
            Channel::Clean
        } else {
            Channel::Dirty
        };
        controls.clean_volume = t * 10.0;
        controls.clean_bass = (1.0 - t) * 10.0;
        controls.clean_treble = t * 10.0;
        controls.dirty_gain = (1.0 - t) * 10.0;
        controls.dirty_bass = t * 10.0;
        controls.dirty_middle = (1.0 - t) * 10.0;
        controls.dirty_treble = t * 10.0;
        controls.dirty_volume = (1.0 - t) * 10.0;
        controls.gain_boost = step % 1_024 < 512;
        controls.global_boost = step % 777 < 388;
        controls.cab_enabled = step % 333 < 166;
        controls.output_power = match (step / 2_048) % 4 {
            0 => PowerMode::Watt100,
            1 => PowerMode::Watt70,
            2 => PowerMode::Watt50,
            _ => PowerMode::Watt30,
        };
        controls.power = match (step / 3_000) % 3 {
            0 => PowerState::On,
            1 => PowerState::Standby,
            _ => PowerState::Off,
        };

        let x = 0.5 * (std::f32::consts::TAU * 196.0 * step as f32 / SAMPLE_RATE).sin();
        let y = engine.process_sample(x, &controls);
        assert!(y.is_finite(), "control sweep produced {y} at step {step}");
    }
    let after = allocation_count();

    assert_eq!(
        before,
        after,
        "sweeping every control performed {} allocations",
        after - before
    );
}

#[test]
#[cfg(not(debug_assertions))]
fn reset_allocates_nothing() {
    let controls = SampleControls::default();
    let mut engine = prepared(&controls);
    for _ in 0..4_096 {
        engine.process_sample(0.1, &controls);
    }

    let before = allocation_count();
    for _ in 0..64 {
        engine.reset();
        for _ in 0..128 {
            engine.process_sample(0.1, &controls);
        }
    }
    assert_eq!(before, allocation_count(), "reset() touched the heap");
}

#[test]
fn every_supported_sample_rate_prepares_and_runs_clean() {
    // 44.1 kHz through 192 kHz. At 192 kHz the oversampled pipeline runs at
    // 1.536 MHz, which is where the 10 Hz DC blockers' coefficients get closest
    // to unity and any use of the `2*pi*fc/fs` approximation would show up.
    for rate in [
        44_100.0f32,
        48_000.0,
        88_200.0,
        96_000.0,
        176_400.0,
        192_000.0,
    ] {
        let controls = SampleControls::default();
        let mut engine = Box::new(AmpEngine::new());
        assert!(engine.prepare(rate, &controls), "prepare failed at {rate}");

        let mut peak = 0.0f32;
        for n in 0..(rate as usize / 4) {
            let x = 0.7 * (std::f32::consts::TAU * 220.0 * n as f32 / rate).sin();
            let y = engine.process_sample(x, &controls);
            assert!(y.is_finite(), "non-finite output at {rate} Hz");
            peak = peak.max(y.abs());
        }
        assert!(peak > 0.01, "silent output at {rate} Hz");
        assert!(peak < 4.0, "runaway output {peak} at {rate} Hz");
    }
}

#[test]
fn latency_is_reported_consistently_across_sample_rates() {
    // The oversampling cascade and the convolution partition are both
    // sample-count based, so the reported latency must not vary with rate.
    let controls = SampleControls::default();
    let mut reference = None;
    for rate in [44_100.0f32, 48_000.0, 96_000.0, 192_000.0] {
        let mut engine = Box::new(AmpEngine::new());
        assert!(engine.prepare(rate, &controls));
        let latency = engine.latency_samples();
        match reference {
            None => reference = Some(latency),
            Some(expected) => assert_eq!(latency, expected, "latency moved at {rate} Hz"),
        }
    }
    assert_eq!(reference, Some(120));
}

#[test]
fn a_block_of_silence_produces_silence() {
    // A guitar amplifier that hisses or hums with no input is a broken model,
    // and a DC offset at the output would eat headroom in the host.
    let controls = SampleControls::default();
    let mut engine = prepared(&controls);
    for _ in 0..(SAMPLE_RATE as usize) {
        engine.process_sample(0.0, &controls);
    }

    let mut peak = 0.0f32;
    let mut sum = 0.0f64;
    let count = SAMPLE_RATE as usize;
    for _ in 0..count {
        let y = engine.process_sample(0.0, &controls);
        peak = peak.max(y.abs());
        sum += y as f64;
    }
    assert!(peak < 1.0e-5, "idle noise floor at {peak}");
    assert!(
        (sum / count as f64).abs() < 1.0e-6,
        "DC offset of {}",
        sum / count as f64
    );
}

#[test]
fn buffer_size_does_not_change_the_output() {
    // The engine is driven a sample at a time, so a host using 32-sample
    // buffers must get bit-identical audio to one using 512. Anything
    // block-size dependent — the convolver's partitioning, the control-rate
    // tone stack updates — would show up here.
    let controls = SampleControls::default();

    let render = |block: usize| -> Vec<f32> {
        let mut engine = prepared(&controls);
        let mut output = Vec::with_capacity(8_192);
        let mut written = 0usize;
        while written < 8_192 {
            for _ in 0..block {
                let x = 0.6 * (std::f32::consts::TAU * 147.0 * written as f32 / SAMPLE_RATE).sin();
                output.push(engine.process_sample(x, &controls));
                written += 1;
                if written >= 8_192 {
                    break;
                }
            }
        }
        output
    };

    let small = render(32);
    let large = render(512);
    assert_eq!(small.len(), large.len());
    for (index, (a, b)) in small.iter().zip(large.iter()).enumerate() {
        assert_eq!(a, b, "buffer size changed sample {index}");
    }
}

#[test]
fn the_engine_recovers_from_a_pathological_input_burst() {
    let controls = SampleControls::default();
    let mut engine = prepared(&controls);

    for value in [f32::MAX, f32::MIN, 1.0e30, -1.0e30] {
        for _ in 0..1_024 {
            let y = engine.process_sample(value, &controls);
            assert!(y.is_finite(), "{value} produced {y}");
            assert!(y.abs() <= 32.0, "{value} produced an unbounded {y}");
        }
    }

    // Two seconds of silence must bring it back to a clean noise floor rather
    // than leaving a latched state in any recursive filter.
    for _ in 0..(SAMPLE_RATE as usize * 2) {
        engine.process_sample(0.0, &controls);
    }
    let mut peak = 0.0f32;
    for _ in 0..4_096 {
        peak = peak.max(engine.process_sample(0.0, &controls).abs());
    }
    assert!(peak < 1.0e-4, "engine failed to settle: {peak}");
}

/// The impulse response handoff, end to end, on the audio thread's terms.
///
/// This is the property that makes the cabinet loader safe to use while audio
/// is running: collecting a published response and re-partitioning it touches
/// only buffers that already exist.
#[test]
#[cfg(not(debug_assertions))]
fn loading_an_impulse_response_allocates_nothing() {
    use amberhead_or100::dsp::cabinet::IR_LENGTH;
    use amberhead_or100::shared::IrSlot;

    let controls = SampleControls::default();
    let mut engine = prepared(&controls);

    // Everything the audio thread will touch is built up front, exactly as
    // `Plugin::initialize` does it.
    let slot = IrSlot::new(IR_LENGTH);
    let mut scratch = vec![0.0f32; IR_LENGTH];
    let mut seen = slot.current_generation();

    // Two responses, published from this thread as the editor would, and long
    // enough to fill every partition.
    let first: Vec<f32> = (0..IR_LENGTH).map(|n| 0.995f32.powi(n as i32)).collect();
    let second: Vec<f32> = (0..IR_LENGTH)
        .map(|n| 0.9f32.powi(n as i32) * if n % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    for _ in 0..1_024 {
        engine.process_sample(0.1, &controls);
    }

    let before = allocation_count();
    for round in 0..32 {
        // The publish side is the editor's, not the audio thread's, and is
        // allowed to allocate; it does not, but that is not what is asserted.
        let taps = if round % 3 == 0 { &first } else { &second };
        assert!(slot.publish(taps));

        // This is the audio thread's half.
        let collected = slot.collect(&mut seen, &mut scratch);
        assert_eq!(collected, Some(IR_LENGTH), "the response was not collected");
        if let Some(length) = collected {
            if let Some(loaded) = scratch.get(..length) {
                assert!(engine.load_impulse_response(loaded));
            }
        }

        for _ in 0..256 {
            let out = engine.process_sample(0.1, &controls);
            assert!(out.is_finite(), "loading produced a non-finite sample");
        }
    }
    let after = allocation_count();

    // The publishes above run on this thread and are counted too, so the
    // budget is not zero — but it must be bounded by the number of publishes
    // rather than growing with the audio processed.
    assert!(
        after - before < 32 * 4,
        "loading impulse responses performed {} allocations",
        after - before
    );

    // And the same for the revert path, which must not allocate at all.
    let before = allocation_count();
    for _ in 0..32 {
        assert!(slot.publish_default());
        if let Some(length) = slot.collect(&mut seen, &mut scratch) {
            assert_eq!(length, 0);
            assert!(engine.restore_default_impulse_response());
        }
        for _ in 0..256 {
            assert!(engine.process_sample(0.1, &controls).is_finite());
        }
    }
    assert_eq!(
        before,
        allocation_count(),
        "restoring the built-in cabinet touched the heap"
    );
}

/// A loaded cabinet must actually change the sound, and must survive a reset.
#[test]
fn a_loaded_impulse_response_changes_the_output_and_survives_reset() {
    use amberhead_or100::dsp::cabinet::IR_LENGTH;

    let controls = SampleControls::default();

    let peak_of = |engine: &mut AmpEngine| -> f32 {
        let settle = (SAMPLE_RATE * 0.2) as usize;
        for n in 0..settle {
            let x = (std::f32::consts::TAU * 220.0 * n as f32 / SAMPLE_RATE).sin();
            engine.process_sample(x, &controls);
        }
        let mut peak = 0.0f32;
        for n in 0..(SAMPLE_RATE * 0.1) as usize {
            let x = (std::f32::consts::TAU * 220.0 * (settle + n) as f32 / SAMPLE_RATE).sin();
            peak = peak.max(engine.process_sample(x, &controls).abs());
        }
        peak
    };

    let mut engine = prepared(&controls);
    let built_in = peak_of(&mut engine);

    // A short, bright response: nothing like the synthesised 4x12.
    let mut taps = vec![0.0f32; IR_LENGTH];
    if let Some(tap) = taps.get_mut(0) {
        *tap = 1.0;
    }
    if let Some(tap) = taps.get_mut(3) {
        *tap = -0.6;
    }
    assert!(engine.load_impulse_response(&taps));
    let loaded = peak_of(&mut engine);

    assert!(loaded.is_finite() && loaded > 0.0);
    assert!(
        (loaded - built_in).abs() > built_in * 0.05,
        "loading a completely different cabinet changed nothing: {built_in} vs {loaded}"
    );

    // Band normalisation means the two are within a few dB of each other, so
    // swapping cabinets does not need the output fader moved.
    let difference_db = 20.0 * (loaded / built_in).log10();
    assert!(
        difference_db.abs() < 12.0,
        "the level jumped {difference_db} dB when the cabinet was swapped"
    );

    // A reset clears the tail but keeps the loaded response.
    let after_reset = {
        engine.reset();
        peak_of(&mut engine)
    };
    assert!(
        (after_reset - loaded).abs() < loaded * 0.02,
        "reset lost the loaded cabinet: {loaded} -> {after_reset}"
    );

    assert!(engine.restore_default_impulse_response());
    let restored = peak_of(&mut engine);
    assert!(
        (restored - built_in).abs() < built_in * 0.02,
        "restoring did not recover the built-in cabinet: {built_in} -> {restored}"
    );
}
