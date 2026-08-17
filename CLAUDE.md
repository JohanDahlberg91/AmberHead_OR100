# CLAUDE.md — Orange OR100 VST3 Development Rules

# GENERAL AI WORKING RULES

## Behavioral Mandates
- **Direct Answers:** Provide the code and technical explanation immediately. No pleasantries or meta-announcements.
- **Complete Implementations:** No placeholder comments (`// TODO`), omitted code, or partial functions.
- **Surgical Changes:** Modify only what was requested. Never refactor surrounding untouched code.

## Code Standards
- **Error Handling:** Explicit error checking everywhere. No panics, unhandled promises, or suppressed errors.
- **Typing:** Strict types only. Zero `any` or loose type casting.
- **Dependencies:** Never install new packages/crates without explicit user approval.

## Testing & Quality
- Include unit tests for every new function/endpoint.
- Ensure all outputs are compilable and pass linting/type-checks.

This document governs all code generation, architecture decisions, and refactoring for the **Orange OR100 Modern Reissue Virtual Analog VST3/CLAP** plugin built with **Rust**, **`nih-plug`**, and **`vizia`**.

---

## 1. Non-Negotiable Real-Time Audio Constraints

The `Plugin::process` audio loop runs on a high-priority, real-time operating system thread. Violation of real-time safety causes audio dropouts, priority inversion, or DAW crashes.

* **STRICTLY ZERO HEAP ALLOCATIONS ON AUDIO THREAD:**
  * Never call `Vec::new()`, `vec![]`, `Box::new()`, `String`, `format!`, or any dynamically resizing collection inside `process()`, `process_sample()`, or child DSP calls.
  * All scratch buffers, delay lines, oversampling filters, and lookup tables MUST be pre-allocated in `initialize()` / `prepare_to_play()`.
* **NO LOCKING OR SYSTEM BLOCKING:**
  * Never use `std::sync::Mutex`, `std::sync::RwLock`, file I/O, network calls, DB lookups, or logging (`println!`, `log::info!`) on the audio thread.
  * Thread communication between UI and DSP must use `nih-plug` atomic parameter proxies or lock-free SPSC ring buffers (`rtrb`).
* **PANIC-FREE & ZERO UNWRAPS:**
  * Never use `.unwrap()`, `.expect()`, or slice index operations `arr[i]` that can panic out of bounds inside the DSP loop. Use `.get()`, branchless masking, or bounded iteration (`for sample in buffer.iter_mut()`).
* **DENORMAL & NAN PROTECTION:**
  * Enable Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) in DSP loops.
  * Add soft-clipping or anti-denormal offsets where recursive filters (IIRs, integrators) risk subnormal floating-point numbers.
  * Protect all feedback and division paths against `NaN` and `Inf` to prevent explosive signals that damage monitors.

---

## 2. DSP & Circuit Modeling Standards

* **12AX7 Preamp Triodes:**
  * Do NOT use naive static waveshapers (e.g., plain `tanh(x)`) as a shortcut.
  * Implement the **Norman Koren triode equation** or precomputed 2D lookup tables ($V_{gk}, V_{pk}$) with cubic interpolation.
  * Every triode stage must model dynamic grid-current conduction ($V_{gk} > 0\text{V}$) into the coupling capacitor to reproduce asymmetric bias drift and transient compression.
  * Every gain stage must include a 1st-order high-pass DC blocker ($f_c \approx 10\text{ Hz}$).
* **Polyphase Oversampling ($8\times$):**
  * All non-linear stages (preamp, phase inverter, power tubes) MUST execute inside an $8\times$ oversampled sub-pipeline.
  * Use SIMD-vectorized polyphase half-band FIR/IIR filters to minimize CPU overhead and phase distortion.
  * Downsample back to the host rate before final output or cabinet convolution.
* **Passive Tone Stacks:**
  * Clean (2-Band) and Dirty (3-Band) tone stacks must be modeled as true interactive circuit networks (bilinear transform of the analog admittance matrix or Wave Digital Filter adaptors). Independent parametric EQ bands are strictly prohibited.
* **Power Section & Dynamic Sag:**
  * Model the push-pull EL34 power stage with Long-Tailed Pair (LTP) phase inversion.
  * Implement a dual-time-constant envelope detector (Attack $\sim 8\text{ ms}$, Release $\sim 120\text{ ms}$) tracking output current to dynamically modulate the virtual $B+$ plate voltage rail.

---

## 3. `nih-plug` & `vizia` Architecture Rules

* **Parameter Declarations:**
  * Derive `Params` on parameter structs.
  * Set appropriate smoothing on continuous controls (e.g., $15\text{–}20\text{ ms}$ on Gain, Volume, and EQ) using `FloatParam::with_smoother(SmoothingStyle::Exponential(...))`.
  * Discrete switches (`gain_boost`, `global_boost`, `channel`, `output_power`) must use `BoolParam` or `EnumParam` with zero smoothing.
* **Skeuomorphic GUI Guidelines:**
  * GUI code must reside strictly under `src/gui/` and remain fully decoupled from internal DSP structures.
  * Custom knob and switch widgets in `vizia` must emit `ParamEvent` updates to `nih-plug` without holding audio thread locks.
  * Render the faceplate using clean vector primitives (`vizia` canvas/FemtoVG) adhering to the Orange "Pics Only" aesthetic (ivory background, fluted black knobs, and vector hieroglyphs).

---

## 4. Code Quality & Output Requirements

* **No Incomplete Code / Placeholders:**
  * Do NOT write `// TODO: Implement later`, `// Logic goes here`, or omit array bounds in DSP math. Provide complete, compilable, mathematically verified implementations.
* **SIMD & Data-Oriented Layout:**
  * Cache-align hot DSP state structures (`#[repr(C)]` or `#[repr(align(64))]`).
  * Utilize `wide` (or `std::simd`) to vectorize multi-channel and oversampled buffer calculations.
* **Documentation & Citations:**
  * Document all filter coefficients, circuit component values ($R$, $C$, $L$), and mathematical formulas with inline docstrings linking them back to the amplifier schematic stages.

---

## 5. Verification & Testing Protocol

* Every DSP module must include headless unit tests:
  1. **Impulse Response Test:** Ensure DC stability and bounded output.
  2. **Harmonic Spectrum Test:** Verify that odd/even harmonics emerge correctly under drive.
  3. **Aliasing Test:** Ensure harmonic energy above $20\text{ kHz}$ is attenuated by $\ge 60\text{ dB}$ before downsampling.
* Real-time alloc assertions must be verified using `assert_process_allocs` in debug builds.