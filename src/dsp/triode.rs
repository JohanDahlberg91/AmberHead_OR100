//! 12AX7 preamp triode stage built on the Norman Koren vacuum tube equations.
//!
//! Specification section 2.A and `CLAUDE.md` §2. A stage models, in order:
//!
//! 1. the coupling capacitor and grid-leak network feeding the grid, including
//!    grid-current conduction once `Vgk > 0 V`;
//! 2. the cathode resistor and its bypass capacitor, which self-bias the stage
//!    and set its low-frequency gain shelf;
//! 3. the Koren plate characteristic intersected with the plate-load line;
//! 4. the Miller-capacitance pole; and
//! 5. a 10 Hz first-order DC blocker.
//!
//! # The Koren equations
//!
//! ```text
//! E1 = (Vpk / Kp) * ln(1 + exp(Kp * (1/mu + (Vgk + Voff) / sqrt(Kvb + Vpk^2))))
//! Ip = (E1^Ex / Kg1) * (1 + sgn(E1))
//! ```
//!
//! `E1` is non-negative by construction, so `sgn(E1)` is 1 whenever the tube
//! conducts and the trailing factor is simply 2.
//!
//! # Solving the operating point
//!
//! `Ip` depends on both `Vgk` *and* `Vpk`, while the plate load imposes
//! `Ip = (Vsupply - Vpk) / Ra`. The intersection is found once per stage at
//! `prepare()` time for 1024 grid voltages and stored in a table that is read
//! back with Catmull-Rom cubic interpolation, exactly as permitted by
//! `CLAUDE.md` §2. `Ip` rises monotonically with `Vpk` while the load line
//! falls, so the intersection is unique and plain bisection is unconditionally
//! convergent — no Newton divergence to guard against.
//!
//! Pre-solving the load line offline is both cheaper and more accurate at run
//! time than interpolating a raw 2D `(Vgk, Vpk)` current surface, because the
//! run-time path performs no iteration at all.

use super::denormal::{flush, sanitize};
use super::filters::{DcBlocker, OnePoleLp};

/// Number of entries in the solved plate-voltage table.
const TABLE_SIZE: usize = 1024;
/// Lowest tabulated grid-cathode voltage. Below this the tube is fully cut off
/// and the plate simply sits at the supply rail.
const VGK_MIN: f32 = -28.0;
/// Highest tabulated grid-cathode voltage. Grid conduction clamps the real grid
/// long before this, so the table only has to stay well defined here.
const VGK_MAX: f32 = 4.0;
/// Bisection iterations per table entry. 60 halvings of a 0..500 V bracket
/// resolve the plate voltage to far below `f32` precision.
const BISECTION_STEPS: usize = 60;

/// Static device parameters of a 12AX7 / ECC83 dual triode.
///
/// Values are those given in specification section 2.A.
#[derive(Debug, Clone, Copy)]
pub struct KorenModel {
    /// Amplification factor `mu`.
    pub mu: f64,
    /// Sharpness of the cutoff knee, `Kp`.
    pub kp: f64,
    /// Knee voltage `Kvb`.
    pub kvb: f64,
    /// Exponent `Ex` of the three-halves-power law.
    pub ex: f64,
    /// Current scaling term.
    ///
    /// Specification section 2.A writes the denominator of `Ip` as `K_p` and
    /// lists `K_p = 600`. In Koren's published formulation the exponential
    /// knee term and the current scaling term are two distinct constants —
    /// `Kp` and `Kg1` — and the 12AX7 value for the scaling term is 1060.
    /// Using 1060 here is what makes the model reproduce the datasheet plate
    /// curves: at `Vgk = 0 V, Vpk = 100 V` it yields 0.75 mA, matching the
    /// published 12AX7 characteristic. Using 600 would inflate every plate
    /// current by a factor of 1.77 and push the quiescent point off the
    /// datasheet curves entirely.
    pub kg1: f64,
    /// Grid contact-potential offset `Voff`, -0.5 V for the 12AX7.
    pub v_off: f64,
}

impl Default for KorenModel {
    fn default() -> Self {
        Self {
            mu: 100.0,
            kp: 600.0,
            kvb: 300.0,
            ex: 1.4,
            kg1: 1060.0,
            v_off: -0.5,
        }
    }
}

impl KorenModel {
    /// Evaluates the Koren plate current in amperes.
    ///
    /// The `ln(1 + exp(z))` softplus is evaluated in its numerically stable
    /// form `max(z, 0) + ln(1 + exp(-|z|))`; the naive expression overflows
    /// `f64` for the `z > 700` values that occur at large positive `Vgk`.
    pub fn plate_current(&self, vgk: f64, vpk: f64) -> f64 {
        let vpk = vpk.max(0.0);
        let knee = (self.kvb + vpk * vpk).sqrt();
        let z = self.kp * (1.0 / self.mu + (vgk + self.v_off) / knee);
        let softplus = z.max(0.0) + (-z.abs()).exp().ln_1p();
        let e1 = (vpk / self.kp) * softplus;
        if e1 <= 0.0 {
            0.0
        } else {
            2.0 * e1.powf(self.ex) / self.kg1
        }
    }
}

/// Circuit values surrounding one triode, taken from the OR100 preamp stages.
#[derive(Debug, Clone, Copy)]
pub struct StageCircuit {
    /// Plate load resistor `Ra` in ohms.
    pub plate_resistor: f64,
    /// `B+` supply feeding the plate load, in volts.
    pub supply_voltage: f64,
    /// Cathode bias resistor `Rk` in ohms.
    pub cathode_resistor: f64,
    /// Corner frequency of the cathode bypass network in Hz.
    ///
    /// `fc = 1 / (2*pi*Ck*(Rk || rk))` where `rk ~ 1/gm`. A large bypass cap
    /// puts this below the audio band (full gain everywhere); a small one lifts
    /// it into the midrange and produces the bass-shy, forward voicing of a
    /// cascaded gain stage.
    pub cathode_bypass_hz: f32,
    /// Coupling capacitor `Cc` into this grid, in farads. Specification
    /// section 2.A: 22 nF.
    pub coupling_cap: f32,
    /// Grid leak resistor `Rg` to ground, in ohms. Specification section 2.A:
    /// 1 MΩ.
    pub grid_leak: f32,
    /// Effective series resistance seen by grid current once the grid-cathode
    /// diode conducts, in ohms. Combines the grid stopper with the conducting
    /// diode's dynamic resistance.
    pub grid_conduction_resistance: f32,
    /// Miller-effect pole at the plate, in Hz.
    ///
    /// `fc = 1 / (2*pi*(Ra || rp)*Cmiller)`; for a 100 kΩ load and the ~100 pF
    /// Miller capacitance of a 12AX7 driving another 12AX7 this lands near
    /// 42 kHz.
    pub miller_cutoff_hz: f32,
}

impl StageCircuit {
    /// A conventional 12AX7 gain stage: 100 kΩ plate load off a 300 V rail,
    /// 1.5 kΩ cathode resistor fully bypassed by 22 µF.
    pub const fn classic_gain_stage() -> Self {
        Self {
            plate_resistor: 100_000.0,
            supply_voltage: 300.0,
            cathode_resistor: 1_500.0,
            cathode_bypass_hz: 16.0,
            coupling_cap: 22.0e-9,
            grid_leak: 1.0e6,
            grid_conduction_resistance: 2_200.0,
            miller_cutoff_hz: 42_000.0,
        }
    }

    /// A cascaded overdrive stage: 100 kΩ plate load, 820 Ω cathode resistor
    /// with a small 0.68 µF bypass cap that rolls the low end off below
    /// ~470 Hz, which is what keeps a high-gain channel from turning to mud.
    pub const fn cascade_stage() -> Self {
        Self {
            plate_resistor: 100_000.0,
            supply_voltage: 300.0,
            cathode_resistor: 820.0,
            cathode_bypass_hz: 470.0,
            coupling_cap: 22.0e-9,
            grid_leak: 1.0e6,
            grid_conduction_resistance: 2_200.0,
            miller_cutoff_hz: 38_000.0,
        }
    }
}

/// A single 12AX7 gain stage with grid conduction, self-bias and DC blocking.
///
/// All voltages are in volts referred to the real circuit, not normalized
/// audio units. [`crate::dsp::engine`] owns the conversion at the amplifier
/// input and output.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct Triode {
    model: KorenModel,
    circuit: StageCircuit,

    /// Solved plate voltage as a function of `Vgk`, built by [`Self::prepare`].
    plate_table: [f32; TABLE_SIZE],
    /// `(TABLE_SIZE - 1) / (VGK_MAX - VGK_MIN)`, the table index scale.
    table_scale: f32,

    /// Plate voltage at the quiescent operating point, subtracted so the stage
    /// emits a signal centred on zero.
    quiescent_plate: f32,
    /// Cathode voltage at the quiescent operating point.
    quiescent_cathode: f32,

    /// Cathode voltage, lowpassed by the bypass network.
    cathode: OnePoleLp,
    /// Miller-capacitance pole at the plate.
    miller: OnePoleLp,
    /// 10 Hz inter-stage DC blocker.
    dc_blocker: DcBlocker,

    /// Charge on the coupling capacitor, i.e. the accumulated bias shift.
    grid_charge: f32,
    /// Per-sample decay of `grid_charge` through the grid leak, `exp(-T/(Rg*Cc))`.
    grid_leak_coeff: f32,
    /// Volts of `grid_charge` gained per ampere-sample of grid current, `T/Cc`.
    grid_charge_gain: f32,
}

impl Default for Triode {
    fn default() -> Self {
        Self {
            model: KorenModel::default(),
            circuit: StageCircuit::classic_gain_stage(),
            plate_table: [0.0; TABLE_SIZE],
            table_scale: 1.0,
            quiescent_plate: 0.0,
            quiescent_cathode: 0.0,
            cathode: OnePoleLp::default(),
            miller: OnePoleLp::default(),
            dc_blocker: DcBlocker::default(),
            grid_charge: 0.0,
            grid_leak_coeff: 1.0,
            grid_charge_gain: 0.0,
        }
    }
}

impl Triode {
    /// Creates a stage around the given circuit values.
    pub fn new(circuit: StageCircuit) -> Self {
        Self {
            circuit,
            ..Self::default()
        }
    }

    /// Builds the load-line table and configures every filter for
    /// `sample_rate`, which is the *oversampled* rate.
    ///
    /// This is the only expensive call in the module; it belongs in
    /// `initialize()`, never in `process()`.
    pub fn prepare(&mut self, sample_rate: f32) {
        self.table_scale = (TABLE_SIZE - 1) as f32 / (VGK_MAX - VGK_MIN);
        for (index, entry) in self.plate_table.iter_mut().enumerate() {
            let vgk = VGK_MIN as f64
                + (VGK_MAX - VGK_MIN) as f64 * index as f64 / (TABLE_SIZE - 1) as f64;
            *entry = Self::solve_plate(&self.model, &self.circuit, vgk) as f32;
        }

        let (cathode, plate) = self.solve_quiescent_point();
        self.quiescent_cathode = cathode;
        self.quiescent_plate = plate;

        self.cathode
            .prepare(self.circuit.cathode_bypass_hz, sample_rate);
        self.cathode.preload(self.quiescent_cathode);
        self.miller
            .prepare(self.circuit.miller_cutoff_hz, sample_rate);
        self.dc_blocker.prepare(sample_rate);

        let period = 1.0 / sample_rate;
        let leak_tau = self.circuit.grid_leak * self.circuit.coupling_cap;
        self.grid_leak_coeff = (-period / leak_tau).exp();
        // Loop gain of the grid-conduction feedback path is
        // `grid_charge_gain / grid_conduction_resistance`. Clamping it to 0.5
        // keeps the one-sample-delayed loop unconditionally stable even at
        // absurdly low sample rates; at every rate this plugin actually runs
        // (>= 8 * 44.1 kHz) the unclamped value is already ~0.05.
        let raw_gain = period / self.circuit.coupling_cap;
        self.grid_charge_gain = raw_gain.min(0.5 * self.circuit.grid_conduction_resistance);

        self.reset();
    }

    /// Returns the stage to its quiescent state without rebuilding the table.
    pub fn reset(&mut self) {
        self.cathode.reset();
        self.cathode.preload(self.quiescent_cathode);
        self.miller.reset();
        self.dc_blocker.reset();
        self.grid_charge = 0.0;
    }

    /// Plate voltage at the quiescent point, in volts.
    pub fn quiescent_plate(&self) -> f32 {
        self.quiescent_plate
    }

    /// Grid bias at the quiescent point, in volts (negative).
    pub fn quiescent_bias(&self) -> f32 {
        -self.quiescent_cathode
    }

    /// Intersects the Koren plate characteristic with the load line
    /// `Ip = (Vsupply - Vpk) / Ra` for a fixed `Vgk`, by bisection.
    fn solve_plate(model: &KorenModel, circuit: &StageCircuit, vgk: f64) -> f64 {
        let mut low = 0.0f64;
        let mut high = circuit.supply_voltage;
        for _ in 0..BISECTION_STEPS {
            let mid = 0.5 * (low + high);
            // Monotonically increasing in `mid`: the tube current rises with
            // plate voltage while the load line's current falls.
            let residual = model.plate_current(vgk, mid)
                - (circuit.supply_voltage - mid) / circuit.plate_resistor;
            if residual > 0.0 {
                high = mid;
            } else {
                low = mid;
            }
        }
        0.5 * (low + high)
    }

    /// Finds the self-biased quiescent point, where the cathode voltage and the
    /// plate current are mutually consistent: `Vk = Ip(-Vk) * Rk`.
    ///
    /// Solved by damped fixed-point iteration; the damping factor keeps the
    /// iteration from oscillating around the solution for high-`Rk` stages.
    fn solve_quiescent_point(&self) -> (f32, f32) {
        let mut cathode = 1.0f64;
        for _ in 0..512 {
            let plate = Self::solve_plate(&self.model, &self.circuit, -cathode);
            let current = (self.circuit.supply_voltage - plate) / self.circuit.plate_resistor;
            let target = current * self.circuit.cathode_resistor;
            cathode += 0.25 * (target - cathode);
        }
        let plate = Self::solve_plate(&self.model, &self.circuit, -cathode);
        (cathode as f32, plate as f32)
    }

    /// Reads the solved plate table with Catmull-Rom cubic interpolation.
    ///
    /// Every index is saturated into range, so no slice access here can panic
    /// regardless of the incoming grid voltage (`CLAUDE.md` §1).
    #[inline(always)]
    fn plate_voltage(&self, vgk: f32) -> f32 {
        let clamped = vgk.clamp(VGK_MIN, VGK_MAX);
        let position = (clamped - VGK_MIN) * self.table_scale;
        let base = position as usize;
        let fraction = position - base as f32;

        let last = TABLE_SIZE - 1;
        let i1 = base.min(last);
        let i0 = i1.saturating_sub(1);
        let i2 = (i1 + 1).min(last);
        let i3 = (i1 + 2).min(last);

        let p0 = self.plate_table.get(i0).copied().unwrap_or(0.0);
        let p1 = self.plate_table.get(i1).copied().unwrap_or(0.0);
        let p2 = self.plate_table.get(i2).copied().unwrap_or(0.0);
        let p3 = self.plate_table.get(i3).copied().unwrap_or(0.0);

        // Catmull-Rom basis, tension 0.5.
        let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
        let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
        let c = -0.5 * p0 + 0.5 * p2;
        ((a * fraction + b) * fraction + c) * fraction + p1
    }

    /// Processes one sample at the oversampled rate.
    ///
    /// `input` is the grid drive voltage arriving through the coupling
    /// capacitor; the return value is the plate signal voltage, DC-blocked and
    /// referred to the quiescent plate voltage. The stage inverts, as a real
    /// common-cathode triode does: a rising grid draws more current and pulls
    /// the plate down.
    #[inline(always)]
    pub fn process(&mut self, input: f32) -> f32 {
        // 1. Coupling capacitor and grid leak. `grid_charge` is the DC shift
        //    accumulated by grid-current rectification, which is what makes a
        //    hard-picked note bloom and then compress.
        let grid = input - self.grid_charge;
        let vgk = grid - self.cathode.value();

        // 2. Grid-current conduction for Vgk > 0 (specification section 2.A).
        let grid_current = if vgk > 0.0 {
            vgk / self.circuit.grid_conduction_resistance
        } else {
            0.0
        };
        self.grid_charge =
            flush(self.grid_charge * self.grid_leak_coeff + grid_current * self.grid_charge_gain);

        // 3. Plate load line, pre-solved.
        let plate = self.plate_voltage(vgk);
        let plate_current =
            (self.circuit.supply_voltage as f32 - plate) / self.circuit.plate_resistor as f32;

        // 4. Cathode self-bias, filtered by the bypass capacitor. The cathode
        //    voltage is fed back with a one-sample delay, which is what lets
        //    the load line be solved offline; at 384 kHz that delay is 2.6 µs
        //    against a bypass time constant of milliseconds.
        self.cathode
            .process(plate_current * self.circuit.cathode_resistor as f32);

        // 5. Miller pole, then DC blocking.
        let signal = self.miller.process(plate - self.quiescent_plate);
        sanitize(self.dc_blocker.process(signal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 8x of a 48 kHz host rate, the rate every preamp stage actually runs at.
    const OS_RATE: f32 = 384_000.0;

    fn prepared_stage(circuit: StageCircuit) -> Triode {
        let mut stage = Triode::new(circuit);
        stage.prepare(OS_RATE);
        stage
    }

    #[test]
    fn koren_current_matches_the_12ax7_datasheet() {
        let model = KorenModel::default();
        // Published 12AX7 plate characteristic: ~0.75 mA at Vg = 0, Vp = 100 V.
        let current = model.plate_current(0.0, 100.0);
        assert!(
            (current - 0.75e-3).abs() < 0.1e-3,
            "Ip was {} mA",
            current * 1e3
        );
        // Cut off hard at a deeply negative grid.
        assert!(model.plate_current(-20.0, 250.0) < 1.0e-9);
        // Monotonic in both arguments.
        assert!(model.plate_current(-1.0, 250.0) > model.plate_current(-2.0, 250.0));
        assert!(model.plate_current(-1.0, 250.0) > model.plate_current(-1.0, 150.0));
    }

    #[test]
    fn quiescent_point_lands_in_a_plausible_class_a_window() {
        let stage = prepared_stage(StageCircuit::classic_gain_stage());
        let bias = stage.quiescent_bias();
        let plate = stage.quiescent_plate();
        // A 1.5 kΩ-biased 12AX7 off 300 V sits around -1.3 V grid / 170..230 V
        // plate. Anything outside that is a broken solver, not a design choice.
        assert!((-2.5..=-0.7).contains(&bias), "bias solved to {bias} V");
        assert!(
            (150.0..=250.0).contains(&plate),
            "plate solved to {plate} V"
        );
    }

    #[test]
    fn plate_table_is_monotonically_decreasing_in_vgk() {
        let stage = prepared_stage(StageCircuit::classic_gain_stage());
        // More positive grid -> more current -> lower plate voltage.
        let mut previous = f32::INFINITY;
        for index in 0..TABLE_SIZE {
            let value = stage.plate_table.get(index).copied().unwrap_or(0.0);
            assert!(value <= previous + 1e-3, "table rose at index {index}");
            previous = value;
        }
    }

    #[test]
    fn cubic_interpolation_agrees_with_the_direct_solve() {
        let stage = prepared_stage(StageCircuit::classic_gain_stage());
        let model = KorenModel::default();
        let circuit = StageCircuit::classic_gain_stage();
        let mut worst = 0.0f32;
        // Sample off-grid points across the active region.
        for step in 0..400 {
            let vgk = -8.0 + 8.5 * step as f32 / 400.0;
            let exact = Triode::solve_plate(&model, &circuit, vgk as f64) as f32;
            worst = worst.max((stage.plate_voltage(vgk) - exact).abs());
        }
        assert!(worst < 0.05, "worst interpolation error {worst} V");
    }

    #[test]
    fn impulse_response_is_bounded_and_settles_to_dc_zero() {
        let mut stage = prepared_stage(StageCircuit::classic_gain_stage());
        let first = stage.process(1.0);
        assert!(first.is_finite());
        let mut peak = first.abs();
        let mut last = first;
        for _ in 0..(OS_RATE as usize) {
            last = stage.process(0.0);
            peak = peak.max(last.abs());
            assert!(last.is_finite());
        }
        assert!(peak < 32.0, "impulse peaked at {peak} V");
        assert!(last.abs() < 1.0e-2, "did not settle, residual {last} V");
    }

    #[test]
    fn small_signal_gain_is_in_the_12ax7_range() {
        let mut stage = prepared_stage(StageCircuit::classic_gain_stage());
        let mut peak = 0.0f32;
        for n in 0..(OS_RATE as usize / 10) {
            let x = 0.01 * (std::f32::consts::TAU * 1_000.0 * n as f32 / OS_RATE).sin();
            let y = stage.process(x);
            if n > OS_RATE as usize / 20 {
                peak = peak.max(y.abs());
            }
        }
        let gain = peak / 0.01;
        // A 12AX7 gain stage with a bypassed cathode gives roughly 50..70x.
        assert!(
            (40.0..=90.0).contains(&gain),
            "small-signal gain was {gain}"
        );
    }

    /// Harmonic levels of a 500 Hz tone relative to the fundamental, in dB.
    ///
    /// A Hann window is applied over the measurement region; without it the
    /// spectral leakage of a non-integer number of periods swamps the -50 dB
    /// harmonics this stage produces at low drive.
    fn harmonics_db(drive: f32, orders: &[u32]) -> Vec<f64> {
        let mut stage = prepared_stage(StageCircuit::classic_gain_stage());
        let f0 = 500.0f64;
        let total = 32_768usize;
        let mut samples = Vec::with_capacity(total);
        for i in 0..total {
            let x = drive as f64 * (std::f64::consts::TAU * f0 * i as f64 / OS_RATE as f64).sin();
            samples.push(stage.process(x as f32) as f64);
        }

        let start = total / 2;
        let length = total - start;
        let magnitude = |frequency: f64| -> f64 {
            let (mut real, mut imag) = (0.0, 0.0);
            for (i, sample) in samples.iter().enumerate().skip(start) {
                let t = (i - start) as f64 / length as f64;
                let window = 0.5 - 0.5 * (std::f64::consts::TAU * t).cos();
                let phase = std::f64::consts::TAU * frequency * i as f64 / OS_RATE as f64;
                real += sample * window * phase.cos();
                imag += sample * window * phase.sin();
            }
            (real * real + imag * imag).sqrt() / length as f64
        };

        let fundamental = magnitude(f0).max(1.0e-12);
        orders
            .iter()
            .map(|order| 20.0 * (magnitude(f0 * *order as f64) / fundamental).log10())
            .collect()
    }

    #[test]
    fn light_drive_produces_mostly_second_harmonic() {
        // `CLAUDE.md` §5 harmonic spectrum test, low-drive case. A triode's
        // curvature is asymmetric, so a signal that never reaches either
        // limit produces almost purely even-order product.
        let levels = harmonics_db(0.5, &[2, 3]);
        let second = levels.first().copied().unwrap_or(f64::NEG_INFINITY);
        let third = levels.get(1).copied().unwrap_or(f64::NEG_INFINITY);
        assert!(second > -35.0, "2nd harmonic only {second} dB");
        assert!(
            second > third + 15.0,
            "expected strong even-order dominance: {second} vs {third} dB"
        );
    }

    #[test]
    fn overdrive_generates_both_orders_with_even_dominance() {
        // High-drive case: the stage now clips against cutoff on one side and
        // grid conduction on the other, so odd orders appear too — but the
        // asymmetry keeps the second harmonic on top.
        let levels = harmonics_db(3.0, &[2, 3, 4]);
        let second = levels.first().copied().unwrap_or(f64::NEG_INFINITY);
        let third = levels.get(1).copied().unwrap_or(f64::NEG_INFINITY);
        let fourth = levels.get(2).copied().unwrap_or(f64::NEG_INFINITY);
        assert!(second > -15.0, "2nd harmonic only {second} dB");
        assert!(third > -25.0, "3rd harmonic only {third} dB");
        assert!(fourth > -25.0, "4th harmonic only {fourth} dB");
        assert!(
            second > third,
            "expected even-order dominance: {second} vs {third} dB"
        );
    }

    #[test]
    fn distortion_grows_monotonically_with_drive() {
        let mut previous = f64::NEG_INFINITY;
        for drive in [0.2f32, 0.5, 1.5, 3.0, 8.0] {
            let second = harmonics_db(drive, &[2])
                .first()
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            assert!(
                second > previous,
                "2nd harmonic fell at drive {drive}: {previous} -> {second} dB"
            );
            previous = second;
        }
    }

    #[test]
    fn grid_conduction_shifts_the_bias_and_recovers() {
        let mut stage = prepared_stage(StageCircuit::classic_gain_stage());
        assert_eq!(stage.grid_charge, 0.0);

        // A hot transient drives the grid positive and rectifies onto the cap.
        for n in 0..2_000 {
            let x = 6.0 * (std::f32::consts::TAU * 200.0 * n as f32 / OS_RATE).sin();
            stage.process(x);
        }
        let charged = stage.grid_charge;
        assert!(charged > 0.5, "bias shifted only {charged} V");

        // With the drive gone it must leak away through the 1 MΩ grid resistor.
        for _ in 0..(OS_RATE as usize / 10) {
            stage.process(0.0);
        }
        assert!(
            stage.grid_charge < charged * 0.05,
            "bias failed to recover, {} V left",
            stage.grid_charge
        );
    }

    #[test]
    fn output_stays_finite_under_absurd_input() {
        let mut stage = prepared_stage(StageCircuit::cascade_stage());
        for x in [1.0e6f32, -1.0e6, f32::MAX, f32::MIN] {
            let y = stage.process(x);
            assert!(y.is_finite(), "input {x} produced {y}");
            assert!(y.abs() <= 32.0);
        }
        // And it must still work normally afterwards.
        for _ in 0..1_000 {
            assert!(stage.process(0.0).is_finite());
        }
    }

    #[test]
    fn cascade_stage_rolls_off_lows_relative_to_mids() {
        // The 0.68 µF bypass cap should make 100 Hz quieter than 1 kHz.
        let level_at = |frequency: f32| -> f32 {
            let mut stage = prepared_stage(StageCircuit::cascade_stage());
            let mut peak = 0.0f32;
            let total = (OS_RATE as usize) / 4;
            for n in 0..total {
                let x = 0.01 * (std::f32::consts::TAU * frequency * n as f32 / OS_RATE).sin();
                let y = stage.process(x);
                if n > total / 2 {
                    peak = peak.max(y.abs());
                }
            }
            peak
        };
        let low = level_at(100.0);
        let mid = level_at(1_000.0);
        assert!(low < mid * 0.85, "low {low} was not below mid {mid}");
    }
}
