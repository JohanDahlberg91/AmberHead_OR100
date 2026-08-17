# Technical Specification & Roadmap: Orange OR100 Modern Reissue VST3

---

## 1. Executive Summary & Scope

This specification defines the architecture, mathematical models, UI layout, and engineering milestones for developing a virtual analog recreation of the **Orange OR100 Modern Reissue (OR100H)** tube amplifier. 

The software targets **VST3**, **CLAP**, and **Standalone** plugin formats for 64-bit Windows, Linux, and macOS platforms. It is implemented natively in **Rust** using the **`nih-plug`** framework for DAW hosting and **`vizia`** (via `nih_plug_vizia`) for a hardware-accelerated vector UI.

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DAW Host (VST3 / CLAP)                                   │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
│
┌──────────────────────────────────────────────▼──────────────────────────────────────────────┐
│                               RUST PLUGIN SHELL (nih-plug)                                  │
│  • AudioIOLayout (Mono In, Mono/Stereo Out)       • Atomic / Smoothed Parameter Matrix      │
│  • State Persistence & Preset Serializer (Serde)   • Realtime-Safe Context & Alloc Assertions│
└──────────────────────┬──────────────────────────────────────────────┬───────────────────────┘
│ Audio Thread (Zero Alloc)                    │ UI Thread (OpenGL/Metal)
┌──────────────────────▼────────────────────────┐   ┌─────────────────▼───────────────────────┐
│              OR100 DSP ENGINE                 │   │         VIZIA SKEUOMORPHIC GUI          │
│  • 8x Polyphase Half-Band Oversampling        │   │  • Ivory Textured Faceplate             │
│  • Cascaded Koren 12AX7 & Grid Drift          │   │  • 8 Animated Knobs & Pics-Only Icons   │
│  • WDF / IIR 2-Band & 3-Band Tone Stacks      │   │  • Metal Toggles (Boosts, Channel, Pwr) │
│  • Push-Pull EL34 & Dynamic B+ Sag Envelope   │   │  • Reactive Pilot Jewel Lamp            │
│  • Partitioned FFT Impulse Response Convolver │   │  • Real-time Preset / State Binding     │
└───────────────────────────────────────────────┘   └─────────────────────────────────────────┘


---

## 2. Audio Processing Architecture & DSP Math

The DSP pipeline executes with zero memory allocations inside the real-time audio thread. All non-linear stages run within an **$8\times$ polyphase oversampling wrapper** to eliminate intermodulation and aliasing distortion.

Audio In ──► [ 8x Upsample ] ──► [ Channel Switch ]
│
┌──────────────────────────────────┴───────────────────────────────────┐
▼ (Clean)                                                              ▼ (Dirty)
[ Stage V1 (12AX7) ]                                                  [ Stage V2 (12AX7) ]
│                                                                      │
[ 2-Band Passive EQ ]                                                 [ Stage V3 (12AX7) ]
│                                                                      │
[ Clean Volume ]                                                      [ Stage V4 (Gain Boost) ]
│                                                                      │
│                                                                [ 3-Band Passive EQ ]
│                                                                      │
│                                                                [ Dirty Volume ]
│                                                                      │
└──────────────────────────────────┬───────────────────────────────────┘
│
[ Global Boost (+3dB) ]
│
[ LTP Phase Inverter ]
│
[ 4x/2x EL34 Push-Pull Stage ] ◄─── [ Dynamic B+ Sag Tracker ]
│
[ 8x Downsample ]
│
[ Output Transformer Core ]
│
[ Partitioned FFT Cab IR ] ──► Audio Out


### A. Vacuum Tube Modeling (12AX7 Preamp Triodes)
Each triode stage models asymmetrical plate voltage clipping and dynamic grid-current conduction using the Norman Koren vacuum tube equation:

$$E_1 = \frac{V_{pk}}{k_p} \ln\left(1 + \exp\left(k_p \left[\frac{1}{\mu} + \frac{V_{gk} + V_{off}}{\sqrt{K_{vb} + V_{pk}^2}}\right]\right)\right)$$

$$I_p = \frac{E_1^{E_x}}{K_p} \left(1 + \text{sgn}(E_1)\right)$$

* **Triode Parameters (12AX7):** $\mu = 100$, $K_p = 600$, $K_{vb} = 300$, $E_x = 1.4$, $V_{off} = -0.5\text{ V}$.
* **Grid Conduction & Bias Drift:** When $V_{gk} > 0\text{V}$, grid diode conduction engages, charging the virtual AC coupling capacitor ($C_c = 22\text{ nF}$, $R_g = 1\text{ M}\Omega$). This dynamic shift in operating bias reproduces the low-end bloom and compression signature during heavy palm muting.
* **DC Blocking:** A 1st-order high-pass filter ($f_c = 10\text{ Hz}$) follows each gain stage to remove DC bias accumulation.

### B. Passive Tone Stack Equations
* **Clean Channel (2-Band):** Dual RC shelf network modeled as a 2nd-order discrete IIR filter. The `Middle` band is fixed; boosting `Bass` dynamically broadens the midrange scoop.
* **Dirty Channel (3-Band):** Interactive Marshall/Orange passive tone stack modeled via bilinear transform of the circuit admittance matrix:
  $$H(s) = \frac{b_3 s^3 + b_2 s^2 + b_1 s + b_0}{a_3 s^3 + a_2 s^2 + a_1 s + 1}$$
  Coefficients are dynamically recalculated on parameter changes to ensure that adjusting `Middle` alters the loading and corner frequencies of `Bass` and `Treble`.

### C. Phase Inverter, Power Stage & Sag Matrix
* **Long-Tailed Pair (LTP):** Generates differential drive voltages $V_{drive}^+$ and $V_{drive}^-$ with second-harmonic asymmetry.
* **Push-Pull EL34 Stage:** Symmetrical power pentode clipping with crossover distortion:
  $$V_{out} = \tanh\left(\frac{V_{drive}^+}{V_{sag}}\right) - \tanh\left(\frac{V_{drive}^-}{V_{sag}}\right)$$
* **Dynamic $B+$ Sag Tracker:** Tracks total power-stage current draw using a dual-time-constant envelope detector:
  $$V_{sag}(t) = V_{nominal} - \Delta V \cdot \text{Env}(|I_{out}|)$$
  * *Attack Time:* $8\text{ ms}$ (initial sag compression).
  * *Release Time:* $120\text{ ms}$ (low-frequency voltage bounce).
* **Wattage Switching Matrix:**
  * **100W Mode:** 4 tubes active, $V_{nominal} = 480\text{V}$.
  * **70W Mode:** 4 tubes active, stepped-down plate voltage $V_{nominal} = 340\text{V}$.
  * **50W Mode:** 2 tubes active (inner pair disabled), $V_{nominal} = 480\text{V}$.
  * **30W Mode:** 2 tubes active, stepped-down plate voltage $V_{nominal} = 340\text{V}$.

---

## 3. Parameter Schema & State Management

Parameters derive `nih_plug::params::Params` with sample-accurate smoothing:

| Parameter Identifier | Display Name | Type | Range / Steps | Default | Smoothing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `channel` | Channel Select | `EnumParam` | Clean (0), Dirty (1) | Dirty | Discrete |
| `clean_volume` | Clean Volume | `FloatParam` | $0.0 \dots 10.0$ (Exp) | $5.0$ | $15\text{ ms}$ |
| `clean_bass` | Clean Bass | `FloatParam` | $0.0 \dots 10.0$ (Lin) | $5.0$ | $20\text{ ms}$ |
| `clean_treble` | Clean Treble | `FloatParam` | $0.0 \dots 10.0$ (Lin) | $5.0$ | $20\text{ ms}$ |
| `dirty_gain` | Dirty Gain | `FloatParam` | $0.0 \dots 10.0$ (Exp) | $6.5$ | $15\text{ ms}$ |
| `dirty_bass` | Dirty Bass | `FloatParam` | $0.0 \dots 10.0$ (Lin) | $5.0$ | $20\text{ ms}$ |
| `dirty_middle` | Dirty Middle | `FloatParam` | $0.0 \dots 10.0$ (Lin) | $5.0$ | $20\text{ ms}$ |
| `dirty_treble` | Dirty Treble | `FloatParam` | $0.0 \dots 10.0$ (Lin) | $5.0$ | $20\text{ ms}$ |
| `dirty_volume` | Dirty Volume | `FloatParam` | $0.0 \dots 10.0$ (Exp) | $5.0$ | $15\text{ ms}$ |
| `gain_boost` | Gain Boost | `BoolParam` | Off, On | Off | Discrete |
| `global_boost` | Global Boost (+3dB)| `BoolParam` | Off, On | Off | $10\text{ ms}$ |
| `power_switch` | Power / Standby | `EnumParam` | Off, Standby, On | On | Discrete |
| `output_power` | Power Mode | `EnumParam` | 100W, 70W, 50W, 30W | 100W | Discrete |
| `cab_enabled` | Cabinet Emulation | `BoolParam` | Bypass, Active | Active | Discrete |

---

## 4. UI Specification (`vizia`)

* **Dimensions:** $920 \times 340\text{ px}$ (Vector scalable from $75\%$ to $200\%$).
* **Aesthetic:** Skeuomorphic Orange "Pics Only" faceplate.
  * **Chassis Background:** Textured cream/ivory enamel panel with top and bottom Orange-stripe framing.
  * **Glyphs:** Crisp vector renders of official icons (Speaker for Volume, Clefs for Bass/Treble, Soundwave for Middle, Fist/Burst for Gain).
  * **Knobs:** Custom-drawn `Vizia` rotary widgets representing fluted black pointer knobs with white indicator lines ($270^\circ$ sweep).
  * **Toggles:** 3-way vertical metal switches for Power (Full/Standby/Half) and 2-way bat switches for Boosts and Channels.
  * **Jewel Lamp:** Dynamic amber pilot light with real-time glow intensity tied to the $B+$ voltage rail state.

---

## 5. Non-Functional Requirements & Performance Budget

* **CPU Target:** $\le 2.0\%$ single-core usage (Core i7 / Apple M1 or equivalent) at $48\text{ kHz}$ buffer sizes down to $32\text{ samples}$.
* **Latency:** Zero reported DSP algorithmic latency (when using minimum-phase polyphase IIR downsampling filters) or bounded linear-phase FIR latency reported to the DAW via `context.set_latency_samples()`.
* **Real-time Safety:** Verified via `nih_plug::assert_process_allocs` in debug mode to prevent heap allocations, lock contention, or blocking I/O on the audio thread.

---

## 6. Milestone Roadmap

┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          MILESTONE ROADMAP                                             │
├─────────────────┬─────────────────┬──────────────────┬─────────────────┬───────────────────────────────┤
│ Phase 1 (W1-W2) │ Phase 2 (W3-W4) │ Phase 3 (W5-W6)  │ Phase 4 (W7-W8) │ Phase 5 (W9-W10)              │
│ Core DSP Engine │ nihil-plug Host │ Vizia GUI Engine │ Cab IR & Polish │ Testing, QA & Multi-Platform  │
└─────────────────┴─────────────────┴──────────────────┴─────────────────┴───────────────────────────────┘


### Phase 1: Core DSP Engine & Prototyping (Weeks 1–2)
* Implement the Norman Koren 12AX7 triode math with dynamic grid conduction and DC blocking in a standalone Rust DSP module.
* Implement the $8\times$ polyphase half-band oversampling engine (SIMD-accelerated via `wide` / `std::simd`).
* Derive and implement discrete transfer functions for both the Clean 2-band and Dirty 3-band interactive tone stacks.
* Implement the LTP phase inverter, push-pull EL34 power stage, and dual-time-constant $B+$ sag envelope.

### Phase 2: `nih-plug` Integration & Parameter Infrastructure (Weeks 3–4)
* Scaffold the Cargo workspace and configure `nih-plug` VST3, CLAP, and Standalone entrypoints.
* Define all 14 parameters with exponential/linear scaling curves and smoothing times.
* Connect the audio buffer loop to the DSP core with channel switching and wattage attenuation logic.
* Run integration tests using `assert_process_allocs` to ensure zero real-time audio thread allocations.

### Phase 3: Skeuomorphic GUI in `vizia` (Weeks 5–6)
* Construct the faceplate layout in `nih_plug_vizia` with responsive CSS styling.
* Implement custom vector knob and toggle switch controls with mouse-drag delta scaling and double-click reset.
* Render the "Pics Only" vector glyphs and the responsive amber jewel pilot lamp.
* Bind GUI events directly to plugin parameter proxies with bidirectional state synchronization.

### Phase 4: Cabinet Simulation & Audio Polish (Weeks 7–8)
* Implement a partitioned, zero-latency FFT convolution engine (`realfft`) for loading impulse responses.
* Embed a default 4x12 Celestion Vintage 30 cabinet IR with a bypass toggle.
* Calibrate gain staging across all wattage modes (100W, 70W, 50W, 30W) against physical reference measurements.
* Refine $B+$ sag recovery curves to nail the low-end transient bloom.

### Phase 5: Verification, Benchmarking & Release Packaging (Weeks 9–10)
* Execute DAW validation suite tests in REAPER, Ableton Live, Cubase, and Bitwig across Windows, macOS, and Linux.
* Profile CPU cache locality and SIMD execution under low-latency conditions (32/64 sample buffers).
* Set up automated GitHub Actions CI/CD workflows utilizing `cargo xtask bundle` to package signed `.vst3`, `.clap`, and standalone binary installers.