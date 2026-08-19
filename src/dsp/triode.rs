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

use super::denormal::{flush, sanitize_volts};
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
    /// Current scaling term `Kg1`.
    ///
    /// Specification section 2.A writes the denominator of `Ip` as `K_p` and
    /// lists `K_p = 600`. In Koren's published formulation the exponential
    /// knee term and the current scaling term are two distinct constants —
    /// `Kp` and `Kg1` — and the 12AX7 value for the scaling term is 1060.
    /// Using 600 in the denominator would inflate every plate current by a
    /// factor of 1.77 and push the quiescent point off the datasheet curves
    /// entirely. See [`KorenModel::default`] for the measured verification.
    pub kg1: f64,
    /// Grid contact-potential offset `Voff`, in volts, added to `Vgk`.
    ///
    /// Zero for the 12AX7: see [`KorenModel::default`]. The term is retained
    /// because Koren's general form admits it and other tube types are fitted
    /// with a non-zero value, but for a device whose parameters were fitted
    /// against *measured* plate curves the contact potential is already baked
    /// into the fit, and adding it again double-counts it.
    pub v_off: f64,
}

impl Default for KorenModel {
    /// Koren's published 12AX7 / ECC83 parameter set.
    ///
    /// # Verification against the RCA 12AX7 datasheet
    ///
    /// The datasheet gives two characteristic operating points. Evaluating
    /// this parameter set at both, and differentiating the model numerically
    /// for `gm = dIp/dVgk` and `rp = dVpk/dIp`:
    ///
    /// | Point | Quantity | Datasheet | Model | Error |
    /// | :--- | :--- | ---: | ---: | ---: |
    /// | `Vpk = 250 V, Vgk = -2 V` | `Ip` | 1.20 mA | 0.95 mA | -21 % |
    /// | | `gm` | 1600 µS | 1670 µS | +4 % |
    /// | | `rp` | 62.5 kΩ | 53.7 kΩ | -14 % |
    /// | | `mu` | 100 | 90 | -10 % |
    /// | `Vpk = 100 V, Vgk = -1 V` | `Ip` | 0.50 mA | 0.10 mA | -80 % |
    ///
    /// The 250 V point — the one a preamp triode actually operates near — is
    /// reproduced to within the spread between individual tubes. The 100 V
    /// point is the known weakness of the Koren form: it under-predicts badly
    /// at low plate voltage and low current, near cutoff. That region is
    /// where the stage is already clipping into cutoff, so the error moves the
    /// exact shape of the cutoff knee rather than the operating point or the
    /// gain.
    ///
    /// # Why `v_off` is zero
    ///
    /// An earlier revision carried `v_off = -0.5 V`, on the reasoning that a
    /// real triode has a contact potential of roughly half a volt. Evaluated
    /// against the datasheet that offset is simply wrong, because Koren's
    /// constants were fitted to measured curves that already contain the
    /// contact potential:
    ///
    /// | `Voff` | `Ip` at 250 V / -2 V | `gm` there | Stage `Ia` | Stage `Va` |
    /// | ---: | ---: | ---: | ---: | ---: |
    /// | 0.0 V | 0.95 mA | 1670 µS | 0.98 mA | 202 V |
    /// | -0.5 V | 0.34 mA | 811 µS | 0.82 mA | 218 V |
    /// | +0.5 V | 1.99 mA | 2455 µS | — | — |
    ///
    /// against a datasheet 1.20 mA / 1600 µS, and against the 100 kΩ / 1.5 kΩ
    /// / 300 V preamp stage that a real Orange or Marshall front end measures
    /// at roughly 1 mA and 200 V on the plate. Zero is the only value of the
    /// three that lands on both.
    fn default() -> Self {
        Self {
            mu: 100.0,
            kp: 600.0,
            kvb: 300.0,
            ex: 1.4,
            kg1: 1060.0,
            v_off: 0.0,
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
    /// The OR100's standard 12AX7 gain stage, read off the factory schematic
    /// (Orange OR 100, print HH A03057).
    ///
    /// 220 kΩ plate load (`M22`) off the 320 V preamp node, 2.4 kΩ cathode
    /// resistor fully bypassed by 50 µF. Both triodes of the first ECC83 are
    /// wired this way; the second differs only in carrying a 60 µF bypass.
    ///
    /// The 220 kΩ load is what makes an Orange preamp stage hotter than the
    /// 100 kΩ a Marshall or Fender front end uses: it lands the stage at 70x
    /// against their ~60x, with a lower quiescent plate (184 V) and so less
    /// room before the plate bottoms out.
    ///
    /// The supply is the node behind the 33 kΩ/10 kΩ dropping chain that feeds
    /// the preamp from the main rail; its 32 µF reservoir is rated 385 V, so
    /// the node sits below that.
    ///
    /// The bypass corner follows from the cap and the impedance at the
    /// cathode: `fc = 1 / (2*pi*(Rk || 1/gm)*Ck)`, and at this operating point
    /// `gm ~ 1.6 mA/V`, so `Rk || 1/gm = 2k4 || 625 = 496 Ω` and 50 µF puts
    /// the corner at 6.4 Hz — below the audio band, i.e. full gain everywhere.
    pub const fn classic_gain_stage() -> Self {
        Self {
            plate_resistor: 220_000.0,
            supply_voltage: 320.0,
            cathode_resistor: 2_400.0,
            cathode_bypass_hz: 6.4,
            coupling_cap: 22.0e-9,
            grid_leak: 1.0e6,
            grid_conduction_resistance: 2_200.0,
            miller_cutoff_hz: 42_000.0,
        }
    }

    /// The driver stage that feeds the phase inverter, from the same schematic.
    ///
    /// 390 kΩ plate load (`M39`) and a 1 kΩ cathode resistor. The schematic
    /// bypasses that resistor through the front panel's 50 kΩ `Boost` pot in
    /// series with 0.1 µF, which only bypasses above about 4 kHz — a top-end
    /// emphasis rather than a gain control — so the corner here is set for the
    /// bypassed case and the pot's position is not modelled.
    ///
    /// This is the hottest stage in the amplifier: the 390 kΩ load pulls the
    /// quiescent plate down to 98 V and the bias to -0.57 V, for 84x of gain
    /// and barely half a volt of grid headroom. It is meant to clip, and in
    /// the original it clips *hard*, because the cathodyne inverter it feeds
    /// has no gain of its own and the EL34 grids still need tens of volts.
    pub const fn driver_stage() -> Self {
        Self {
            plate_resistor: 390_000.0,
            supply_voltage: 320.0,
            cathode_resistor: 1_000.0,
            cathode_bypass_hz: 4_100.0,
            coupling_cap: 22.0e-9,
            grid_leak: 1.0e6,
            grid_conduction_resistance: 2_200.0,
            miller_cutoff_hz: 38_000.0,
        }
    }

    /// The extra cascade stage the dirty channel adds over the schematic's
    /// three, kept on the schematic's 220 kΩ plate load but with a smaller
    /// bypass capacitor.
    ///
    /// The factory circuit trims the low end once, at the input, with the
    /// `Depth` rotary switch; a channel with a fourth stage in front of the
    /// tone stack has to trim it again or the cascade turns to mud. A 0.68 µF
    /// bypass on the 2.4 kΩ cathode puts the corner at 470 Hz, so the stage
    /// runs at full gain through the midrange and sheds the bottom octaves
    /// before they reach the next grid.
    pub const fn cascade_stage() -> Self {
        Self {
            plate_resistor: 220_000.0,
            supply_voltage: 320.0,
            cathode_resistor: 2_400.0,
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
        // Circuit volts, not normalized audio: a stage driven into cutoff
        // presents its whole plate swing to the next grid, so this is bounded
        // by the rail rather than by an audio-level ceiling.
        sanitize_volts(self.dc_blocker.process(signal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::denormal::VOLTAGE_LIMIT;

    /// 8x of a 48 kHz host rate, the rate every preamp stage actually runs at.
    const OS_RATE: f32 = 384_000.0;

    fn prepared_stage(circuit: StageCircuit) -> Triode {
        let mut stage = Triode::new(circuit);
        stage.prepare(OS_RATE);
        stage
    }

    /// Numerical transconductance `gm = dIp/dVgk` at an operating point, in
    /// siemens.
    fn transconductance(model: &KorenModel, vgk: f64, vpk: f64) -> f64 {
        let h = 1.0e-4;
        (model.plate_current(vgk + h, vpk) - model.plate_current(vgk - h, vpk)) / (2.0 * h)
    }

    /// Numerical plate resistance `rp = dVpk/dIp` at an operating point, in
    /// ohms.
    fn plate_resistance(model: &KorenModel, vgk: f64, vpk: f64) -> f64 {
        let h = 1.0e-2;
        (2.0 * h) / (model.plate_current(vgk, vpk + h) - model.plate_current(vgk, vpk - h))
    }

    #[test]
    fn koren_current_matches_the_12ax7_datasheet_at_the_preamp_operating_point() {
        let model = KorenModel::default();
        // RCA 12AX7 published characteristic: Vp = 250 V, Vg = -2 V gives
        // Ip = 1.2 mA, gm = 1600 uS, rp = 62.5 kOhm, mu = 100. This is the
        // point a preamp triode actually sits near, so it is the point the
        // model has to reproduce. See `KorenModel::default` for the measured
        // comparison table and for why the low-plate-voltage point is not
        // asserted here.
        let current = model.plate_current(-2.0, 250.0);
        assert!(
            (0.85e-3..=1.35e-3).contains(&current),
            "Ip at 250 V / -2 V was {} mA, datasheet 1.20 mA",
            current * 1e3
        );

        let gm = transconductance(&model, -2.0, 250.0);
        assert!(
            (1.30e-3..=1.90e-3).contains(&gm),
            "gm was {} uS, datasheet 1600 uS",
            gm * 1e6
        );

        let rp = plate_resistance(&model, -2.0, 250.0);
        assert!(
            (48.0e3..=72.0e3).contains(&rp),
            "rp was {} kOhm, datasheet 62.5 kOhm",
            rp / 1e3
        );

        // mu = gm * rp must recover the 12AX7's defining figure of merit.
        let mu = gm * rp;
        assert!((85.0..=110.0).contains(&mu), "mu came out at {mu}");
    }

    #[test]
    fn koren_model_carries_no_contact_potential_offset() {
        // Regression guard. `v_off` was once -0.5 V, which cut the plate
        // current at the datasheet operating point from 0.95 mA to 0.34 mA
        // against a published 1.20 mA, and halved the transconductance. The
        // 12AX7 constants are fitted to measured curves, which already contain
        // the contact potential.
        let model = KorenModel::default();
        assert_eq!(model.v_off, 0.0);

        let shifted = KorenModel {
            v_off: -0.5,
            ..KorenModel::default()
        };
        let reference = 1.20e-3;
        let error_of = |m: &KorenModel| (m.plate_current(-2.0, 250.0) - reference).abs();
        assert!(
            error_of(&model) < error_of(&shifted),
            "the offset model tracked the datasheet better, which cannot be right"
        );
    }

    #[test]
    fn koren_current_is_cut_off_and_monotonic() {
        let model = KorenModel::default();
        // Cut off hard at a deeply negative grid.
        assert!(model.plate_current(-20.0, 250.0) < 1.0e-9);
        // Monotonic in both arguments.
        assert!(model.plate_current(-1.0, 250.0) > model.plate_current(-2.0, 250.0));
        assert!(model.plate_current(-1.0, 250.0) > model.plate_current(-1.0, 150.0));
        // A tube cannot conduct with no plate voltage.
        assert_eq!(model.plate_current(0.0, 0.0), 0.0);
        assert_eq!(model.plate_current(0.0, -10.0), 0.0);
    }

    #[test]
    fn quiescent_point_matches_a_measured_preamp_stage() {
        let circuit = StageCircuit::classic_gain_stage();
        let stage = prepared_stage(circuit);
        let bias = stage.quiescent_bias();
        let plate = stage.quiescent_plate();
        let current =
            (circuit.supply_voltage as f32 - plate) / circuit.plate_resistor as f32 * 1.0e3;

        // The OR100's 220 kOhm / 2.4 kOhm stage off its ~320 V preamp node.
        // A 220 kOhm load draws about half the current of the 100 kOhm one a
        // Marshall front end uses and idles lower on the load line: the model
        // lands at 0.62 mA / 184 V / 1.48 V, and the cathode voltage it solves
        // for is self-consistent with the 2.4 kOhm resistor (0.62 mA * 2k4 =
        // 1.48 V), which is the check that matters — the bias is not asserted
        // independently of the current that produces it.
        assert!((-1.7..=-1.3).contains(&bias), "bias solved to {bias} V");
        assert!(
            (172.0..=196.0).contains(&plate),
            "plate solved to {plate} V"
        );
        assert!(
            (0.52..=0.72).contains(&current),
            "plate current solved to {current} mA"
        );
        let cathode_from_current = current * 1.0e-3 * circuit.cathode_resistor as f32;
        assert!(
            (cathode_from_current + bias).abs() < 0.02,
            "bias {bias} V disagrees with {cathode_from_current} V across Rk"
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
        // 0.1 V sits comfortably inside the stage's 1.46 V bias window, so
        // this measures the impulse response of the amplifier rather than the
        // response of `sanitize`'s clamp. Behaviour under absurd input is
        // covered by `output_stays_finite_under_absurd_input`.
        let first = stage.process(0.1);
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
        //
        // 8 V is what it takes to engage *both* limits on a stage biased at
        // -1.46 V: at 3 V only the grid-conduction side is clamped, and the
        // one-sided clipping that produces is almost purely even-order
        // (2nd -8.8 dB, 3rd -36.8 dB). Odd content is evidence of symmetric
        // clipping, so the drive has to be high enough to reach cutoff too.
        let levels = harmonics_db(8.0, &[2, 3, 4]);
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
            // Bounded by the containment limit, not by an audio-level ceiling:
            // this node carries circuit volts. The plate table itself is
            // bracketed by the 300 V rail, so the observed excursion here is
            // ~78 V — the DC blocker's transient response to a step, not a
            // runaway.
            assert!(y.abs() <= VOLTAGE_LIMIT, "input {x} produced {y} V");
        }
        // And it must still work normally afterwards.
        for _ in 0..1_000 {
            assert!(stage.process(0.0).is_finite());
        }
    }

    #[test]
    fn plate_swing_is_not_capped_at_the_normalized_audio_ceiling() {
        // Regression guard: the stage output was once passed through the
        // normalized-audio `sanitize()`, which brick-wall clipped every plate
        // at ±32 V. That is a third of this stage's real swing, and cascading
        // two of them meant the second stage could never be driven past its
        // own clipping point no matter where the gain pot sat.
        let mut stage = prepared_stage(StageCircuit::cascade_stage());
        let mut peak = 0.0f32;
        let total = OS_RATE as usize / 10;
        for n in 0..total {
            let x = 5.0 * (std::f32::consts::TAU * 500.0 * n as f32 / OS_RATE).sin();
            let y = stage.process(x);
            if n > total / 2 {
                peak = peak.max(y.abs());
            }
        }
        // A stage biased at -1.03 V off a 300 V rail, slammed with 5 V, swings
        // most of the way between cutoff and saturation.
        assert!(peak > 100.0, "hard-driven plate only reached {peak} V");
        assert!(peak < 300.0, "plate exceeded its own supply: {peak} V");
    }

    #[test]
    fn stage_operating_points_match_the_factory_schematic() {
        // Values read off the Orange OR 100 schematic, print HH A03057, run
        // through the Koren model. These are the numbers the gain staging in
        // `crate::dsp::engine` is calibrated against, so a change to either
        // stage's components should fail here before it silently rebalances
        // the whole amplifier.
        let gain_stage = prepared_stage(StageCircuit::classic_gain_stage());
        assert!(
            (-1.6..=-1.35).contains(&gain_stage.quiescent_bias()),
            "220k/2k4 stage biased at {} V",
            gain_stage.quiescent_bias()
        );
        assert!(
            (175.0..=195.0).contains(&gain_stage.quiescent_plate()),
            "220k/2k4 stage idles at {} V",
            gain_stage.quiescent_plate()
        );

        // The driver is the hot one: a 390 kΩ load pulls its plate down near
        // 100 V and leaves it barely half a volt of grid headroom.
        let driver = prepared_stage(StageCircuit::driver_stage());
        assert!(
            (-0.7..=-0.45).contains(&driver.quiescent_bias()),
            "390k/1k driver biased at {} V",
            driver.quiescent_bias()
        );
        assert!(
            driver.quiescent_plate() < gain_stage.quiescent_plate() - 60.0,
            "driver idles at {} V, not far below the gain stage's {} V",
            driver.quiescent_plate(),
            gain_stage.quiescent_plate()
        );
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
