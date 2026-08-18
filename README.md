# AmberHead OR100

A virtual analog recreation of the **Orange OR100 Modern Reissue** guitar amplifier,
written in Rust with [`nih-plug`](https://github.com/robbert-vdh/nih-plug) and
[`vizia`](https://github.com/vizia/vizia).

Builds as **VST3**, **CLAP** and a **standalone** application for 64-bit Windows,
macOS and Linux.

## Building

```sh
# Run the full verification suite (see "Testing" below for why release).
cargo test --release

# Produce target/bundled/{AmberHead OR100.vst3, .clap, standalone}
cargo xtask bundle amberhead_or100 --release
```

Linux needs the usual audio and GUI development headers:

```sh
sudo apt-get install libasound2-dev libgl-dev libjack-jackd2-dev libx11-xcb-dev \
  libxcb1-dev libxcb-dri2-0-dev libxcb-icccm4-dev libxcursor-dev \
  libxkbcommon-dev libxcb-shape0-dev libxcb-xfixes0-dev
```

## Signal chain

Everything from the channel switch through the power tubes runs inside an **8x
polyphase oversampled** sub-pipeline. The output transformer and cabinet run at
the host rate.

```
In -> [8x Up] -> [Channel Switch]
        Clean: V1 ---------------> 2-band EQ -> Clean Volume
        Dirty: V2 -> V3 -> V4 ---> 3-band EQ -> Dirty Volume
      -> [Global Boost +3 dB] -> [LTP Phase Inverter]
      -> [Push-Pull EL34] <--- [Dynamic B+ Sag Tracker]
      -> [8x Down] -> [Output Transformer Core] -> [Partitioned FFT Cab IR] -> Out
```

| Stage | Model |
| :--- | :--- |
| 12AX7 triodes (six of them) | Norman Koren plate equations, load line pre-solved into a 1024-entry table read with Catmull-Rom cubic interpolation. Grid-current conduction charges the coupling capacitor for bias drift; self-biasing cathode network; Miller pole; 10 Hz DC blocker. |
| Oversampling | Three cascaded 2x Kaiser-windowed polyphase half-band FIRs (100/80/70 dB stopbands), dot products vectorised with `wide::f32x8`. |
| Tone stacks | Real circuit networks. The Fender/Marshall passive topology is stamped into a nodal admittance matrix, capacitors are discretised with trapezoidal (bilinear) companion models, and the 6x6 system is reduced at control rate into a 4x4 matrix the audio loop applies in 16 multiply-adds. |
| Phase inverter | Long-tailed pair built from two full triode models, with tail imbalance. |
| Power stage | Push-pull EL34 with a class-AB crossover notch and a dual-time-constant (8 ms / 120 ms) B+ sag tracker feeding the four-mode wattage matrix. |
| Output transformer | Flux integrator with a cubic core-reluctance term (odd-order low-frequency saturation) plus a leakage-inductance resonance. |
| Cabinet | Uniformly-partitioned FFT convolution (`realfft`), 1024-tap IR, 64-sample partitions. |

## Controls

Fourteen automatable parameters. Volume and gain controls use an exponential
range with 15 ms smoothing; the tone controls are linear with 20 ms smoothing;
the switches are discrete.

| Parameter | Type | Range | Default |
| :--- | :--- | :--- | :--- |
| `channel` | enum | Clean, Dirty | Dirty |
| `clean_volume`, `clean_bass`, `clean_treble` | float | 0.0 - 10.0 | 5.0 |
| `dirty_gain` | float | 0.0 - 10.0 | 6.5 |
| `dirty_bass`, `dirty_middle`, `dirty_treble`, `dirty_volume` | float | 0.0 - 10.0 | 5.0 |
| `gain_boost` | bool | Off, On | Off |
| `global_boost` | bool | Off, On (+3 dB) | Off |
| `power_switch` | enum | Off, Standby, On | On |
| `output_power` | enum | 100 W, 70 W, 50 W, 30 W | 100 W |
| `cab_enabled` | bool | Bypass, Active | Active |

The knob pointer tracks the **plain** value rather than the normalized one, so a
control reading `5.0` points at the middle of its travel even on an
exponentially skewed range.

Shift-drag for fine adjustment, double-click or right-click to reset, scroll to
step.

## Latency

The plugin reports **120 samples** to the host, constant across sample rates and
across the cabinet bypass toggle:

| Source | Host samples |
| :--- | ---: |
| 8x oversampling cascade (linear-phase half-band FIRs) | 56 |
| Cabinet convolution partition | 64 |

The oversampler's exact group delay is 56.25 samples; the reported figure is
rounded, leaving a quarter-sample residual well below any DAW's delay
compensation resolution.

## Real-time safety

`Plugin::process` performs **no heap allocation, no locking and no I/O**. Every
lookup table, filter design, tone stack solution, delay line and FFT plan is
built in `initialize()`.

This is verified two ways:

- `nih-plug`'s `assert_process_allocs` feature is enabled, which traps
  allocations inside a hosted debug build.
- `tests/realtime_safety.rs` installs a thread-local counting global allocator
  and asserts the counter does not move across a second of audio, across a sweep
  of every control, or across `reset()`. The suite includes a check that the
  counter itself works, so broken instrumentation cannot produce a false pass.

Flush-to-zero and denormals-are-zero are set for the duration of the audio
callback via MXCSR and restored on drop; every recursive filter additionally
flushes its own state in software, for targets where that guard is unavailable.

## Testing

```sh
cargo test --release      # 155 tests
cargo clippy --all-targets -- -D warnings
```

Release mode is not optional: the DSP tests push millions of oversampled samples
through the model, and the real-time safety suite only compiles in release,
because `nih-plug` owns the global allocator in debug builds.

Beyond the usual unit coverage, the suite verifies the properties that actually
matter for a modelled amplifier:

- **Aliasing** - a hard-clipped 4 kHz tone produces no non-harmonic energy above
  -60 dB, and stage 1's half-band stopband is verified below -90 dB.
- **Harmonic structure** - the Koren triode produces even-order-dominant
  distortion at low drive and both orders under overdrive, growing monotonically
  with drive.
- **Tone stack interaction** - moving `Middle` measurably changes the 100 Hz
  response and moving `Bass` changes the 600 Hz response, which an
  implementation built from independent parametric bands could not do.
- **Sag** - the rail drops under sustained drive, recovers over its release
  constant, and the two-tube modes sag less than the four-tube ones.
- **Determinism** - output is bit-identical at 32-sample and 512-sample host
  buffer sizes.
- **Stability** - `f32::MAX`, infinities and NaN inputs produce bounded finite
  output, and the chain settles back to silence afterwards.

## Deviations from the specification

Made deliberately, with the reasoning recorded in the source:

1. **The cabinet IR is synthesised, not sampled.** `TECH_SPEC.md` asks for an
   embedded Celestion Vintage 30 impulse. Redistributing a real measurement
   would mean redistributing someone else's copyrighted recording, so
   `dsp::cabinet::synthesise_4x12_ir` builds the response from a documented
   filter cascade: driver resonance, cone breakup, the off-axis presence notch,
   voice-coil rolloff and an early cabinet reflection. It is regenerated per
   sample rate, which a fixed 48 kHz WAV could not be.
2. **Koren current scaling uses Kg1 = 1060, not Kp = 600.** The spec writes
   `K_p` in the denominator of `I_p`, but in Koren's formulation the exponential
   knee constant and the current scaling constant are distinct. 1060 is the
   published 12AX7 value and is what makes the model return 0.75 mA at
   Vgk = 0 V, Vpk = 100 V, matching the datasheet curve; 600 would inflate every
   plate current by a factor of 1.77.
3. **The clean channel's tone stack is 3rd order, not 2nd.** `CLAUDE.md`
   section 2 forbids modelling tone stacks as anything but true interactive
   circuit networks, and the real 2-band circuit has three capacitors. It is
   modelled as the same network with the mid element fixed at 15 kOhm.
4. **The wattage selector is a 4-position lever.** Specification section 4
   describes 3-way switches, but the switching matrix in section 2.C defines
   four modes, which a 3-position switch cannot address.
5. **Faceplate glyphs are original vector constructions.** They convey the same
   meanings as Orange's pictograms but are not traced from their artwork.
6. **The release profile does not abort on panic.** Cargo cannot link
   integration tests against an abort-strategy rlib while the test harness
   unwinds, so enabling it would make the verification suite unbuildable.

## Licence

GPL-3.0-or-later, as required by the VST3 SDK bindings `nih-plug` uses.

*Orange* and *OR100* are trademarks of Orange Amplification. This project is an
independent, unaffiliated recreation and ships no Orange artwork, firmware or
audio recordings.
