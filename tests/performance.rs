//! CPU cost measurements against the budget in specification section 5:
//! **≤ 2.0 % of one core at 48 kHz, at buffer sizes down to 32 samples**.
//!
//! At 48 kHz that budget is 20 ms of CPU per second of audio, or **417 ns per
//! host sample**. Everything here reports against that figure.
//!
//! These are measurements, not assertions about behaviour, so they are
//! `#[ignore]`d and stay out of the normal suite. Run them with:
//!
//! ```text
//! cargo test --release --test performance -- --ignored --nocapture
//! ```
//!
//! `--release` is not optional. A debug build runs this DSP more than an order
//! of magnitude slower and the numbers mean nothing.
//!
//! # Method
//!
//! Every measurement runs a warm-up pass, then takes the **minimum** of
//! several timed repeats. Minimum rather than mean: the workload is
//! deterministic, so run-to-run variation is scheduler noise and thermal
//! throttling, all of which only ever adds time. Outputs are accumulated and
//! passed through [`std::hint::black_box`] so the optimiser cannot delete the
//! work being timed.
//!
//! # Clock drift, and why the breakdown interleaves
//!
//! On a laptop the sustained clock can be half the boost clock, and the whole
//! suite is long enough to cross that boundary: measuring the same work at the
//! start and the end of a run has produced figures a factor of two apart here.
//! Absolute numbers are therefore a **range**, not a point, and should be read
//! as such.
//!
//! Relative cost is what drives decisions, so [`cost_breakdown_by_stage`] does
//! not measure each stage to completion in turn — that would charge whichever
//! stage happened to run last for the machine having warmed up. It round-robins
//! across all stages on every repeat, so drift lands on all of them equally.

use std::hint::black_box;
use std::time::{Duration, Instant};

use amberhead_or100::dsp::cabinet::Cabinet;
use amberhead_or100::dsp::engine::{AmpEngine, Channel, PowerState, SampleControls};
use amberhead_or100::dsp::oversampling::Oversampler8x;
use amberhead_or100::dsp::power::{PhaseInverter, PowerAmp, PowerMode};
use amberhead_or100::dsp::tonestack::{ToneStack, ToneStackCircuit};
use amberhead_or100::dsp::transformer::OutputTransformer;
use amberhead_or100::dsp::triode::{StageCircuit, Triode};
use amberhead_or100::dsp::OVERSAMPLING_FACTOR;

/// Budget from specification section 5, as a fraction of one core.
const CPU_BUDGET: f64 = 0.02;
/// Repeats per measurement; the fastest wins.
const REPEATS: usize = 3;

/// Runs `body` `REPEATS` times over `samples` items and returns the fastest
/// per-item cost in nanoseconds.
fn time_per_item<F: FnMut(usize)>(samples: usize, mut body: F) -> f64 {
    // Warm up caches, branch predictors and the CPU's clock ramp.
    for i in 0..samples.min(4_096) {
        body(i);
    }

    let mut best = Duration::MAX;
    for _ in 0..REPEATS {
        let start = Instant::now();
        for i in 0..samples {
            body(i);
        }
        best = best.min(start.elapsed());
    }
    best.as_secs_f64() * 1.0e9 / samples as f64
}

/// Length of the pre-rendered excitation table. 4096 floats is 16 kB, which
/// stays inside L1 alongside a triode's 4 kB load-line table.
const EXCITATION_LEN: usize = 4_096;

/// A test signal that keeps every stage working: loud enough to saturate the
/// cascade, and built from three inharmonic partials so no stage sees a
/// trivially periodic input.
///
/// It is rendered **once**, outside every timed region. Generating it inline
/// costs three `sin` calls per sample, which is the same order as a triode and
/// swamped the per-stage figures when this harness first ran.
fn excitation_table(sample_rate: f32) -> Vec<f32> {
    (0..EXCITATION_LEN)
        .map(|index| {
            let t = index as f32 / sample_rate;
            0.4 * (std::f32::consts::TAU * 110.0 * t).sin()
                + 0.3 * (std::f32::consts::TAU * 277.0 * t).sin()
                + 0.2 * (std::f32::consts::TAU * 1_567.0 * t).sin()
        })
        .collect()
}

fn prepared_engine(controls: &SampleControls, sample_rate: f32) -> Box<AmpEngine> {
    let mut engine = Box::new(AmpEngine::new());
    assert!(
        engine.prepare(sample_rate, controls),
        "engine failed to prepare at {sample_rate} Hz"
    );
    engine
}

/// Percentage of one core a per-sample cost represents at `sample_rate`.
fn core_percent(ns_per_sample: f64, sample_rate: f64) -> f64 {
    100.0 * ns_per_sample * sample_rate / 1.0e9
}

#[test]
#[ignore = "measurement, not a behavioural test; run with --release --ignored"]
fn full_chain_against_the_two_percent_budget() {
    println!(
        "\nbudget: {:.1} % of one core  ->  {:.0} ns per sample at 48 kHz\n",
        CPU_BUDGET * 100.0,
        CPU_BUDGET * 1.0e9 / 48_000.0
    );
    println!(
        "{:<34} {:>10} {:>10} {:>12}",
        "configuration", "ns/sample", "% of core", "headroom"
    );

    for rate in [48_000.0f32, 96_000.0, 192_000.0] {
        for (label, controls) in [
            (
                "dirty, cab on",
                SampleControls {
                    dirty_gain: 8.0,
                    ..SampleControls::default()
                },
            ),
            (
                "dirty, cab bypassed",
                SampleControls {
                    dirty_gain: 8.0,
                    cab_enabled: false,
                    ..SampleControls::default()
                },
            ),
            (
                "clean, cab on",
                SampleControls {
                    channel: Channel::Clean,
                    ..SampleControls::default()
                },
            ),
        ] {
            let mut engine = prepared_engine(&controls, rate);
            let signal = excitation_table(rate);
            let mut accumulator = 0.0f32;
            let samples = (rate * 0.5) as usize;
            let ns = time_per_item(samples, |i| {
                let x = signal[i % EXCITATION_LEN];
                accumulator += engine.process_sample(black_box(x), black_box(&controls));
            });
            black_box(accumulator);

            let percent = core_percent(ns, rate as f64);
            let headroom = CPU_BUDGET * 100.0 / percent;
            println!(
                "{:<34} {ns:>10.1} {percent:>9.2} % {headroom:>10.1}x",
                format!("{} kHz  {label}", rate as u32 / 1000)
            );
        }
    }
    println!();
}

#[test]
#[ignore = "measurement, not a behavioural test; run with --release --ignored"]
fn cost_breakdown_by_stage() {
    const RATE: f32 = 48_000.0;
    const ITEMS: usize = 200_000;
    let oversampled = RATE * OVERSAMPLING_FACTOR as f32;

    let signal = &*Box::leak(excitation_table(oversampled).into_boxed_slice());
    let host_signal = &*Box::leak(excitation_table(RATE).into_boxed_slice());

    // Each entry owns its stage and reports one call. `calls` is how many
    // times the amplifier invokes it per host sample: eight for anything
    // inside the oversampled section, one for the host-rate stages.
    //
    // `instances` is how many of that stage the dirty channel runs. The clean
    // stage and its tone stack are *not* free while the dirty channel is
    // selected: `process_sample` still calls them with a zero input to keep
    // their state settled, so they cost full price.
    struct Stage<'a> {
        label: &'a str,
        instances: u32,
        calls: u32,
        run: Box<dyn FnMut(usize) -> f32 + 'a>,
    }

    let mut gain_stage = Triode::new(StageCircuit::classic_gain_stage());
    gain_stage.prepare(oversampled);
    let mut cascade = Triode::new(StageCircuit::cascade_stage());
    cascade.prepare(oversampled);
    let mut driver = Triode::new(StageCircuit::driver_stage());
    driver.prepare(oversampled);
    let mut stack = ToneStack::new(ToneStackCircuit::or100_dirty());
    stack.prepare(oversampled, 0.5, 0.5, 0.5);
    let mut inverter = PhaseInverter::new();
    inverter.prepare(oversampled);
    let mut power = PowerAmp::new();
    power.prepare(oversampled);
    power.set_mode(PowerMode::Watt100);
    let mut oversampler = Oversampler8x::default();
    oversampler.prepare();
    let mut transformer = OutputTransformer::new();
    transformer.prepare(RATE);
    let mut cabinet = Cabinet::new();
    assert!(cabinet.prepare(RATE));

    let over = OVERSAMPLING_FACTOR as u32;
    let mut stages: Vec<Stage> = vec![
        Stage {
            label: "triode, gain stage (220k/2k4)",
            instances: 2,
            calls: over,
            run: Box::new(move |i| gain_stage.process(black_box(signal[i % EXCITATION_LEN]))),
        },
        Stage {
            label: "triode, cascade stage",
            instances: 3,
            calls: over,
            run: Box::new(move |i| cascade.process(black_box(signal[i % EXCITATION_LEN]))),
        },
        Stage {
            label: "triode, driver stage (390k/1k)",
            instances: 1,
            calls: over,
            run: Box::new(move |i| driver.process(black_box(signal[i % EXCITATION_LEN]))),
        },
        Stage {
            label: "tone stack",
            instances: 2,
            calls: over,
            run: Box::new(move |i| stack.process(black_box(signal[i % EXCITATION_LEN]))),
        },
        Stage {
            label: "phase inverter (2 triodes)",
            instances: 1,
            calls: over,
            run: Box::new(move |i| {
                let pair = inverter.process(black_box(signal[i % EXCITATION_LEN]));
                pair[0] + pair[1]
            }),
        },
        Stage {
            label: "power amp",
            instances: 1,
            calls: over,
            run: Box::new(move |i| {
                let drive = signal[i % EXCITATION_LEN];
                power.process(black_box([drive, -drive]))
            }),
        },
        Stage {
            label: "oversampler up+down cascade",
            instances: 1,
            calls: 1,
            run: Box::new(move |i| {
                oversampler.process(black_box(host_signal[i % EXCITATION_LEN]), |s| s)
            }),
        },
        Stage {
            label: "output transformer",
            instances: 1,
            calls: 1,
            run: Box::new(move |i| transformer.process(black_box(host_signal[i % EXCITATION_LEN]))),
        },
        Stage {
            label: "cabinet (partitioned FFT)",
            instances: 1,
            calls: 1,
            run: Box::new(move |i| {
                cabinet.process(black_box(host_signal[i % EXCITATION_LEN]), true)
            }),
        },
    ];

    // Warm every stage before any of them is timed.
    let mut sink = 0.0f32;
    for stage in stages.iter_mut() {
        for i in 0..4_096 {
            sink += (stage.run)(i);
        }
    }

    // Round-robin: repeat r measures every stage once, so clock drift over the
    // run lands on all of them together rather than on whichever ran last.
    let mut best = vec![Duration::MAX; stages.len()];
    for _ in 0..REPEATS {
        for (index, stage) in stages.iter_mut().enumerate() {
            let start = Instant::now();
            for i in 0..ITEMS {
                sink += (stage.run)(i);
            }
            best[index] = best[index].min(start.elapsed());
        }
    }
    black_box(sink);

    println!("\nper host sample at 48 kHz, stages interleaved across repeats\n");
    println!(
        "{:<34} {:>5} {:>6} {:>11} {:>10} {:>8}",
        "stage", "count", "calls", "ns/sample", "% of core", "share"
    );

    let costs: Vec<f64> = best
        .iter()
        .zip(stages.iter())
        .map(|(duration, stage)| {
            let per_call = duration.as_secs_f64() * 1.0e9 / ITEMS as f64;
            per_call * stage.calls as f64 * stage.instances as f64
        })
        .collect();
    let sum: f64 = costs.iter().sum();

    for (stage, cost) in stages.iter().zip(costs.iter()) {
        println!(
            "{:<34} {:>5} {:>6} {cost:>11.1} {:>8.2} % {:>7.1} %",
            stage.label,
            stage.instances,
            stage.calls,
            core_percent(*cost, 48_000.0),
            100.0 * cost / sum
        );
    }
    println!(
        "\n{:<34} {:>5} {:>6} {sum:>11.1} {:>8.2} %",
        "sum of the parts",
        "",
        "",
        core_percent(sum, 48_000.0)
    );
    println!(
        "\nbudget is {:.0} ns per host sample; this is {:.1}x over it\n",
        CPU_BUDGET * 1.0e9 / 48_000.0,
        sum / (CPU_BUDGET * 1.0e9 / 48_000.0)
    );
}

#[test]
#[ignore = "measurement, not a behavioural test; run with --release --ignored"]
fn buffer_size_does_not_change_the_cost() {
    // The budget is specified "down to 32 samples", so per-buffer overhead
    // matters as much as per-sample cost. The engine has no block-rate work of
    // its own, but the cabinet's partitioned convolution runs on 64-sample
    // partitions, so this checks the cost does not blow up on short buffers.
    const RATE: f32 = 48_000.0;
    let controls = SampleControls {
        dirty_gain: 8.0,
        ..SampleControls::default()
    };

    println!(
        "\n{:<16} {:>10} {:>10}",
        "host buffer", "ns/sample", "% of core"
    );
    for block in [32usize, 64, 128, 256, 512, 1024] {
        let mut engine = prepared_engine(&controls, RATE);
        let signal = excitation_table(RATE);
        let mut accumulator = 0.0f32;
        let blocks = 600;
        let mut index = 0usize;
        let ns_per_block = time_per_item(blocks, |_| {
            for _ in 0..block {
                let x = signal[index % EXCITATION_LEN];
                accumulator += engine.process_sample(black_box(x), black_box(&controls));
                index += 1;
            }
        });
        black_box(accumulator);
        let ns = ns_per_block / block as f64;
        println!(
            "{:<16} {ns:>10.1} {:>8.2} %",
            format!("{block} samples"),
            core_percent(ns, RATE as f64)
        );
    }
    println!();
}

#[test]
#[ignore = "measurement, not a behavioural test; run with --release --ignored"]
fn idle_and_standby_cost() {
    // A muted or standby amplifier should not cost what a working one does;
    // if it does, hosts pay for silent tracks.
    const RATE: f32 = 48_000.0;
    println!("\n{:<28} {:>10} {:>10}", "state", "ns/sample", "% of core");
    for (label, controls, amplitude) in [
        (
            "playing",
            SampleControls {
                dirty_gain: 8.0,
                ..SampleControls::default()
            },
            0.4f32,
        ),
        (
            "silent input",
            SampleControls {
                dirty_gain: 8.0,
                ..SampleControls::default()
            },
            0.0,
        ),
        (
            "standby",
            SampleControls {
                power: PowerState::Standby,
                ..SampleControls::default()
            },
            0.4,
        ),
    ] {
        let mut engine = prepared_engine(&controls, RATE);
        let signal = excitation_table(RATE);
        let mut accumulator = 0.0f32;
        let ns = time_per_item((RATE * 0.5) as usize, |i| {
            let x = amplitude * signal[i % EXCITATION_LEN] / 0.4;
            accumulator += engine.process_sample(black_box(x), black_box(&controls));
        });
        black_box(accumulator);
        println!(
            "{label:<28} {ns:>10.1} {:>8.2} %",
            core_percent(ns, RATE as f64)
        );
    }
    println!();
}
