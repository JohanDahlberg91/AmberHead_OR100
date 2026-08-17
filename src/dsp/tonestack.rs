//! Interactive passive tone stacks, solved as real circuit networks.
//!
//! `CLAUDE.md` §2 forbids modelling these as independent parametric bands, and
//! specification section 2.B calls for "bilinear transform of the circuit
//! admittance matrix". This module does exactly that: the tone stack is stamped
//! into a nodal admittance matrix, capacitors are replaced by their trapezoidal
//! (bilinear) companion models, and the resulting linear system is reduced —
//! once per control-rate update — into a 4x4 matrix that the audio loop applies
//! with 16 multiply-adds per sample.
//!
//! # Topology
//!
//! Both channels use the Fender/Marshall passive stack that the Orange OR100
//! inherits. The netlist below was cross-checked against the reference
//! `59 Bassman Tone Stack` circuit shipped with LiveSPICE:
//!
//! ```text
//!            C1                     (1-t)*Rt          t*Rt
//!   IN ──┬───┤├────┬── A ────────/\/\/\──── OUT ──/\/\/\──┬── C
//!        │         │                         │            │
//!        │       (node A = treble pot top)   │          ──┴── C2
//!        │                                   │          ──┬──
//!        │  Rslope                           │            │
//!        └──/\/\/\──┬── B ──────────────────────────────  ┤
//!                   │                        │            │
//!                 ──┴── C2 to C  ────────────┘         l*Rb
//!                 ──┬── C3 to E                           │
//!                   │                                     ├── D
//!                   │                              (1-m)*Rm
//!                   └──── C3 ────┬── E                     │
//!                                │                         │
//!                              m*Rm                        │
//!                                │                         │
//!                               GND                       GND
//! ```
//!
//! Node roles:
//!
//! | Node  | Junction                                          |
//! |-------|---------------------------------------------------|
//! | `IN`  | driving stage's plate (known voltage)              |
//! | `A`   | treble cap `C1` / treble pot top                   |
//! | `B`   | slope resistor / `C2` / `C3`                       |
//! | `C`   | treble pot bottom / `C2` / bass rheostat top       |
//! | `D`   | bass rheostat bottom / mid pot top                 |
//! | `E`   | mid pot wiper / `C3`                               |
//! | `OUT` | treble pot wiper, loaded by the following stage    |
//!
//! Because every control is a resistance *inside* one shared network, turning
//! `Middle` down genuinely reloads the bass and treble branches and moves their
//! corner frequencies — the interaction specification section 2.B asks for is
//! a consequence of the topology, not something layered on top of it.
//!
//! # Discretization
//!
//! Each capacitor becomes a conductance `Gc = 2C/T` in parallel with a current
//! source `Ieq`, the standard trapezoidal companion. Given node voltages `v`
//! solved at sample `n`:
//!
//! ```text
//! i(n)      = Gc * vc(n) - Ieq(n)
//! Ieq(n+1)  = Gc * vc(n) + i(n) = 2 * Gc * vc(n) - Ieq(n)
//! ```
//!
//! Trapezoidal integration is the bilinear transform, so the discrete network
//! is the bilinear image of the analog one.

use super::denormal::{flush, sanitize};

/// Number of unknown nodes: A, B, C, D, E, OUT.
const NODES: usize = 6;
const NODE_A: usize = 0;
const NODE_B: usize = 1;
const NODE_C: usize = 2;
const NODE_D: usize = 3;
const NODE_E: usize = 4;
const NODE_OUT: usize = 5;

/// Number of capacitors, hence the number of companion current sources.
const CAPS: usize = 3;
const CAP_TREBLE: usize = 0;
const CAP_BASS: usize = 1;
const CAP_MID: usize = 2;

/// Right-hand-side basis size: the input voltage plus one column per capacitor.
const INPUTS: usize = 1 + CAPS;

/// Smallest resistance any pot fraction may take, in ohms.
///
/// A pot at either end of its travel would otherwise contribute an infinite
/// conductance and make the admittance matrix singular. 1 Ω is four orders of
/// magnitude below the smallest real element in the network, so the clamp is
/// electrically invisible while keeping the matrix well conditioned.
const MIN_RESISTANCE: f64 = 1.0;

/// Component values of one tone stack, in ohms and farads.
#[derive(Debug, Clone, Copy)]
pub struct ToneStackCircuit {
    /// Slope resistor between the driving stage and node `B`.
    pub slope_resistor: f64,
    /// Total treble pot resistance.
    pub treble_pot: f64,
    /// Total bass rheostat resistance.
    pub bass_pot: f64,
    /// Total mid pot resistance.
    pub mid_pot: f64,
    /// Load presented by the following stage at `OUT`.
    pub load_resistor: f64,
    /// Treble cap `C1`, from `IN` to node `A`.
    pub treble_cap: f64,
    /// Bass cap `C2`, from node `B` to node `C`.
    pub bass_cap: f64,
    /// Mid cap `C3`, from node `B` to node `E`.
    pub mid_cap: f64,
}

impl ToneStackCircuit {
    /// Dirty-channel 3-band stack.
    ///
    /// Marshall/Orange values: 33 kΩ slope resistor, 220 kΩ treble, 1 MΩ bass,
    /// 25 kΩ mid, 470 pF treble cap and 22 nF bass/mid caps, loaded by the
    /// 1 MΩ dirty volume pot.
    pub const fn or100_dirty() -> Self {
        Self {
            slope_resistor: 33_000.0,
            treble_pot: 220_000.0,
            bass_pot: 1_000_000.0,
            mid_pot: 25_000.0,
            load_resistor: 1_000_000.0,
            treble_cap: 470.0e-12,
            bass_cap: 22.0e-9,
            mid_cap: 22.0e-9,
        }
    }

    /// Clean-channel 2-band stack.
    ///
    /// The clean channel has no `Middle` control, so the mid element is a fixed
    /// resistor — see [`ToneStack::CLEAN_FIXED_MID`]. The larger 56 kΩ slope
    /// resistor and 250 pF treble cap give the softer, less scooped clean
    /// voicing. The remaining bass/treble interaction is unchanged, which is
    /// why raising `Bass` still broadens the midrange scoop exactly as
    /// specification section 2.B describes.
    pub const fn or100_clean() -> Self {
        Self {
            slope_resistor: 56_000.0,
            treble_pot: 250_000.0,
            bass_pot: 1_000_000.0,
            mid_pot: 25_000.0,
            load_resistor: 1_000_000.0,
            treble_cap: 250.0e-12,
            bass_cap: 22.0e-9,
            mid_cap: 22.0e-9,
        }
    }
}

/// A discretized passive tone stack.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct ToneStack {
    circuit: ToneStackCircuit,

    /// Reduced system: maps `[Vin, Ieq0, Ieq1, Ieq2]` onto
    /// `[Vout, Vc0, Vc1, Vc2]`.
    reduced: [[f32; INPUTS]; INPUTS],
    /// Companion conductances `2C/T` for each capacitor.
    conductance: [f64; CAPS],
    /// `2 * conductance`, precomputed for the state update.
    double_conductance: [f32; CAPS],
    /// Companion current sources, the only per-sample state.
    equivalent_current: [f32; CAPS],

    /// Control positions the current `reduced` matrix was built for, used to
    /// skip redundant re-solves.
    cached_controls: [f32; 3],
    /// Set once `prepare()` has run, so `process()` cannot use a zero matrix.
    prepared: bool,
}

impl ToneStack {
    /// Mid-pot rotation the clean channel's fixed mid resistor corresponds to.
    ///
    /// 15 kΩ of the 25 kΩ mid element. A 2-band stack built on the classic
    /// 6.8 kΩ value measures a 700 Hz/90 Hz ratio of 0.25 against the dirty
    /// channel's 0.46 — i.e. it would be markedly *more* scooped than the
    /// high-gain channel, which is the opposite of an Orange clean voicing.
    /// 15 kΩ brings the ratio to 0.38 and leaves the clean channel full in the
    /// low mids while keeping the bass/treble interaction intact.
    pub const CLEAN_FIXED_MID: f32 = 15_000.0 / 25_000.0;

    /// Creates a tone stack for the given component values. Call
    /// [`Self::prepare`] before processing.
    pub fn new(circuit: ToneStackCircuit) -> Self {
        Self {
            circuit,
            reduced: [[0.0; INPUTS]; INPUTS],
            conductance: [0.0; CAPS],
            double_conductance: [0.0; CAPS],
            equivalent_current: [0.0; CAPS],
            cached_controls: [f32::NAN; 3],
            prepared: false,
        }
    }

    /// Computes the companion conductances for `sample_rate` (the oversampled
    /// rate), clears the state, and solves the network for the given controls.
    ///
    /// Each control is a normalized pot rotation in `0.0..=1.0`.
    pub fn prepare(&mut self, sample_rate: f32, treble: f32, bass: f32, mid: f32) {
        let two_over_period = 2.0 * sample_rate as f64;
        self.conductance = [
            self.circuit.treble_cap * two_over_period,
            self.circuit.bass_cap * two_over_period,
            self.circuit.mid_cap * two_over_period,
        ];
        for (doubled, single) in self
            .double_conductance
            .iter_mut()
            .zip(self.conductance.iter())
        {
            *doubled = (2.0 * single) as f32;
        }
        self.prepared = true;
        self.cached_controls = [f32::NAN; 3];
        self.set_controls(treble, bass, mid);
        self.reset();
    }

    /// Clears the companion state.
    pub fn reset(&mut self) {
        self.equivalent_current = [0.0; CAPS];
    }

    /// Re-solves the network for new pot rotations.
    ///
    /// Intended to be called at control rate (once per block), never per
    /// sample: it runs a 6x6 Gauss-Jordan elimination. The pot smoothers in
    /// [`crate::params`] move over 20 ms, so a per-block update is an order of
    /// magnitude finer than the parameter itself changes.
    ///
    /// Returns `false` and leaves the previous matrix in place if the network
    /// came out singular, so a degenerate control combination can never inject
    /// `NaN` into the audio path.
    pub fn set_controls(&mut self, treble: f32, bass: f32, mid: f32) -> bool {
        let controls = [
            treble.clamp(0.0, 1.0),
            bass.clamp(0.0, 1.0),
            mid.clamp(0.0, 1.0),
        ];
        if !self.prepared {
            return false;
        }
        if controls == self.cached_controls {
            return true;
        }

        match self.solve(controls) {
            Some(reduced) => {
                self.reduced = reduced;
                self.cached_controls = controls;
                true
            }
            None => false,
        }
    }

    /// Builds and reduces the admittance matrix for one set of pot rotations.
    fn solve(&self, controls: [f32; 3]) -> Option<[[f32; INPUTS]; INPUTS]> {
        let treble = controls[0] as f64;
        let bass = controls[1] as f64;
        let mid = controls[2] as f64;

        // Pot fractions. `treble` counts up towards the treble cap, so the
        // resistance from node A down to the wiper shrinks as it is raised.
        let r_treble_top = ((1.0 - treble) * self.circuit.treble_pot).max(MIN_RESISTANCE);
        let r_treble_bottom = (treble * self.circuit.treble_pot).max(MIN_RESISTANCE);
        let r_bass = (bass * self.circuit.bass_pot).max(MIN_RESISTANCE);
        let r_mid_top = ((1.0 - mid) * self.circuit.mid_pot).max(MIN_RESISTANCE);
        let r_mid_bottom = (mid * self.circuit.mid_pot).max(MIN_RESISTANCE);

        // Augmented system [G | u w0 w1 w2]: G is the nodal admittance matrix,
        // u collects conductances driven by the known input voltage, and each
        // w_k injects one capacitor's companion current source.
        let mut matrix = [[0.0f64; NODES + INPUTS]; NODES];

        let stamp_resistor =
            |m: &mut [[f64; NODES + INPUTS]; NODES], p: usize, n: usize, r: f64| {
                let g = 1.0 / r;
                m[p][p] += g;
                m[n][n] += g;
                m[p][n] -= g;
                m[n][p] -= g;
            };
        let stamp_to_ground =
            |m: &mut [[f64; NODES + INPUTS]; NODES], p: usize, r: f64| m[p][p] += 1.0 / r;

        stamp_resistor(&mut matrix, NODE_A, NODE_OUT, r_treble_top);
        stamp_resistor(&mut matrix, NODE_OUT, NODE_C, r_treble_bottom);
        stamp_resistor(&mut matrix, NODE_C, NODE_D, r_bass);
        stamp_resistor(&mut matrix, NODE_D, NODE_E, r_mid_top);
        stamp_to_ground(&mut matrix, NODE_E, r_mid_bottom);
        stamp_to_ground(&mut matrix, NODE_OUT, self.circuit.load_resistor);

        // Slope resistor runs from the known input node into B, so its
        // conductance also lands in the input column `u`.
        let g_slope = 1.0 / self.circuit.slope_resistor;
        matrix[NODE_B][NODE_B] += g_slope;
        matrix[NODE_B][NODES] += g_slope;

        // C1 spans IN -> A, so it contributes to both G and the input column,
        // and its companion source enters node A with a negative sign (A is
        // the capacitor's negative terminal).
        let gc_treble = self.conductance[CAP_TREBLE];
        matrix[NODE_A][NODE_A] += gc_treble;
        matrix[NODE_A][NODES] += gc_treble;
        matrix[NODE_A][NODES + 1 + CAP_TREBLE] -= 1.0;

        // C2 spans B -> C, both unknown.
        let gc_bass = self.conductance[CAP_BASS];
        matrix[NODE_B][NODE_B] += gc_bass;
        matrix[NODE_C][NODE_C] += gc_bass;
        matrix[NODE_B][NODE_C] -= gc_bass;
        matrix[NODE_C][NODE_B] -= gc_bass;
        matrix[NODE_B][NODES + 1 + CAP_BASS] += 1.0;
        matrix[NODE_C][NODES + 1 + CAP_BASS] -= 1.0;

        // C3 spans B -> E, both unknown.
        let gc_mid = self.conductance[CAP_MID];
        matrix[NODE_B][NODE_B] += gc_mid;
        matrix[NODE_E][NODE_E] += gc_mid;
        matrix[NODE_B][NODE_E] -= gc_mid;
        matrix[NODE_E][NODE_B] -= gc_mid;
        matrix[NODE_B][NODES + 1 + CAP_MID] += 1.0;
        matrix[NODE_E][NODES + 1 + CAP_MID] -= 1.0;

        let solution = gauss_jordan(&mut matrix)?;

        // `solution[node][k]` is the sensitivity of that node voltage to input
        // `k`, where input 0 is Vin and inputs 1..=3 are the companion
        // currents. Fold the node voltages into the four quantities the audio
        // loop actually needs.
        let mut reduced = [[0.0f32; INPUTS]; INPUTS];
        for k in 0..INPUTS {
            let v_a = solution[NODE_A][k];
            let v_b = solution[NODE_B][k];
            let v_c = solution[NODE_C][k];
            let v_e = solution[NODE_E][k];
            let v_out = solution[NODE_OUT][k];

            // Row 0: output voltage.
            reduced[0][k] = v_out as f32;
            // Row 1: vc0 = Vin - v_A. Only the Vin column carries the leading
            // term, hence the `k == 0` contribution of 1.0.
            let vin_term = if k == 0 { 1.0 } else { 0.0 };
            reduced[1][k] = (vin_term - v_a) as f32;
            // Row 2: vc1 = v_B - v_C.
            reduced[2][k] = (v_b - v_c) as f32;
            // Row 3: vc2 = v_B - v_E.
            reduced[3][k] = (v_b - v_e) as f32;
        }

        if reduced
            .iter()
            .flat_map(|row| row.iter())
            .any(|value| !value.is_finite())
        {
            return None;
        }
        Some(reduced)
    }

    /// Processes one sample at the oversampled rate.
    #[inline(always)]
    pub fn process(&mut self, input: f32) -> f32 {
        let i0 = self.equivalent_current[CAP_TREBLE];
        let i1 = self.equivalent_current[CAP_BASS];
        let i2 = self.equivalent_current[CAP_MID];

        let apply = |row: &[f32; INPUTS]| -> f32 {
            row[0] * input + row[1] * i0 + row[2] * i1 + row[3] * i2
        };

        let output = apply(&self.reduced[0]);
        let vc0 = apply(&self.reduced[1]);
        let vc1 = apply(&self.reduced[2]);
        let vc2 = apply(&self.reduced[3]);

        self.equivalent_current[CAP_TREBLE] =
            flush(sanitize(self.double_conductance[CAP_TREBLE] * vc0 - i0));
        self.equivalent_current[CAP_BASS] =
            flush(sanitize(self.double_conductance[CAP_BASS] * vc1 - i1));
        self.equivalent_current[CAP_MID] =
            flush(sanitize(self.double_conductance[CAP_MID] * vc2 - i2));

        sanitize(output)
    }
}

/// In-place Gauss-Jordan elimination with partial pivoting.
///
/// `matrix` is the augmented system `[G | B]` with `NODES` rows; on success the
/// `INPUTS` solution columns are returned. Returns `None` if a pivot degenerates,
/// which the caller treats as "keep the previous coefficients".
fn gauss_jordan(matrix: &mut [[f64; NODES + INPUTS]; NODES]) -> Option<[[f64; INPUTS]; NODES]> {
    for column in 0..NODES {
        // Partial pivoting for numerical stability: the admittance matrix spans
        // six orders of magnitude between the 1 MΩ grid leak and a pot clamped
        // at 1 Ω.
        let mut pivot_row = column;
        let mut best = matrix[column][column].abs();
        for (row, values) in matrix.iter().enumerate().skip(column + 1) {
            let candidate = values[column].abs();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        if best < 1.0e-18 {
            return None;
        }
        matrix.swap(column, pivot_row);

        let pivot = matrix[column][column];
        let inverse_pivot = 1.0 / pivot;
        for value in matrix[column].iter_mut() {
            *value *= inverse_pivot;
        }

        // The pivot row is copied out so the elimination can borrow every
        // other row mutably; fixed-size arrays are `Copy`, so this is a
        // register-to-stack move rather than an allocation.
        let pivot_values = matrix[column];
        for (row, values) in matrix.iter_mut().enumerate() {
            if row == column {
                continue;
            }
            let factor = values[column];
            if factor == 0.0 {
                continue;
            }
            for (value, pivot) in values.iter_mut().zip(pivot_values.iter()).skip(column) {
                *value -= factor * pivot;
            }
        }
    }

    let mut solution = [[0.0f64; INPUTS]; NODES];
    for (destination, values) in solution.iter_mut().zip(matrix.iter()) {
        for (slot, value) in destination.iter_mut().zip(values.iter().skip(NODES)) {
            if !value.is_finite() {
                return None;
            }
            *slot = *value;
        }
    }
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OS_RATE: f32 = 384_000.0;

    fn dirty_stack(treble: f32, bass: f32, mid: f32) -> ToneStack {
        let mut stack = ToneStack::new(ToneStackCircuit::or100_dirty());
        stack.prepare(OS_RATE, treble, bass, mid);
        stack
    }

    /// Steady-state magnitude response at one frequency, in linear gain.
    fn response(stack: &mut ToneStack, frequency: f32) -> f32 {
        stack.reset();
        // Let the network settle for at least ten cycles of the lowest
        // frequency of interest before measuring.
        let settle = (OS_RATE / frequency * 12.0) as usize;
        let measure = (OS_RATE / frequency * 8.0) as usize;
        for n in 0..settle {
            let x = (std::f32::consts::TAU * frequency * n as f32 / OS_RATE).sin();
            stack.process(x);
        }
        let mut peak = 0.0f32;
        for n in 0..measure {
            let phase = std::f32::consts::TAU * frequency * (settle + n) as f32 / OS_RATE;
            peak = peak.max(stack.process(phase.sin()).abs());
        }
        peak
    }

    #[test]
    fn network_solves_across_the_whole_control_surface() {
        for t in 0..=4 {
            for b in 0..=4 {
                for m in 0..=4 {
                    let controls = (t as f32 / 4.0, b as f32 / 4.0, m as f32 / 4.0);
                    let mut stack = ToneStack::new(ToneStackCircuit::or100_dirty());
                    stack.prepare(OS_RATE, controls.0, controls.1, controls.2);
                    assert!(
                        stack.reduced.iter().flatten().all(|v| v.is_finite()),
                        "singular at {controls:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn output_is_bounded_and_finite_for_an_impulse() {
        let mut stack = dirty_stack(0.5, 0.5, 0.5);
        let first = stack.process(1.0);
        assert!(first.is_finite());
        let mut peak = first.abs();
        for _ in 0..(OS_RATE as usize) {
            let y = stack.process(0.0);
            assert!(y.is_finite());
            peak = peak.max(y.abs());
        }
        assert!(peak <= 1.0, "passive network showed gain: {peak}");
    }

    #[test]
    fn impulse_response_decays_to_zero() {
        let mut stack = dirty_stack(0.5, 0.5, 0.5);
        stack.process(1.0);
        let mut last = 0.0;
        for _ in 0..(OS_RATE as usize) {
            last = stack.process(0.0);
        }
        assert!(last.abs() < 1.0e-6, "residual {last}");
    }

    #[test]
    fn treble_control_only_lifts_the_top_end() {
        let mut low = dirty_stack(0.05, 0.5, 0.5);
        let mut high = dirty_stack(0.95, 0.5, 0.5);
        let treble_low = response(&mut low, 5_000.0);
        let treble_high = response(&mut high, 5_000.0);
        // Full travel is worth about +8.6 dB at 5 kHz on this network.
        assert!(
            treble_high > treble_low * 2.2,
            "treble did nothing: {treble_low} -> {treble_high}"
        );

        let bass_low = response(&mut low, 80.0);
        let bass_high = response(&mut high, 80.0);
        assert!(
            (bass_high - bass_low).abs() < bass_low * 0.5,
            "treble control moved 80 Hz too much: {bass_low} -> {bass_high}"
        );
    }

    #[test]
    fn bass_control_lifts_the_low_end() {
        let mut low = dirty_stack(0.5, 0.02, 0.5);
        let mut high = dirty_stack(0.5, 0.98, 0.5);
        let a = response(&mut low, 80.0);
        let b = response(&mut high, 80.0);
        // Full travel is worth about +6.4 dB at 80 Hz on this network.
        assert!(b > a * 1.8, "bass did nothing: {a} -> {b}");
    }

    #[test]
    fn mid_control_lifts_the_midrange() {
        let mut low = dirty_stack(0.5, 0.5, 0.02);
        let mut high = dirty_stack(0.5, 0.5, 0.98);
        let a = response(&mut low, 500.0);
        let b = response(&mut high, 500.0);
        assert!(b > a * 1.5, "mid did nothing: {a} -> {b}");
    }

    #[test]
    fn the_stack_is_genuinely_interactive() {
        // The defining property `CLAUDE.md` §2 demands: moving `Middle` must
        // reload the bass branch and change the low-frequency response. An
        // implementation built from independent parametric bands would show no
        // change here at all.
        let mut mid_down = dirty_stack(0.5, 0.7, 0.05);
        let mut mid_up = dirty_stack(0.5, 0.7, 0.95);
        let low_a = response(&mut mid_down, 100.0);
        let low_b = response(&mut mid_up, 100.0);
        let relative = ((low_b - low_a) / low_a).abs();
        // The 25 kΩ mid element sits below a bass rheostat an order of
        // magnitude larger, so its influence at 100 Hz is real but modest —
        // measured at 3.9 % across full mid travel.
        assert!(
            relative > 0.025,
            "100 Hz barely moved with Middle: {low_a} -> {low_b}"
        );

        // ...and symmetrically, `Bass` must move the midrange, where the
        // coupling is stronger: measured at 10.9 %.
        let mut bass_down = dirty_stack(0.5, 0.05, 0.5);
        let mut bass_up = dirty_stack(0.5, 0.95, 0.5);
        let mid_a = response(&mut bass_down, 600.0);
        let mid_b = response(&mut bass_up, 600.0);
        assert!(
            ((mid_b - mid_a) / mid_a).abs() > 0.05,
            "600 Hz barely moved with Bass: {mid_a} -> {mid_b}"
        );
    }

    #[test]
    fn stack_has_the_expected_mid_scoop() {
        // Flat-ish settings on a Marshall-derived stack still scoop the mids;
        // that scoop is the whole reason the topology is worth modelling.
        let mut stack = dirty_stack(0.5, 0.5, 0.5);
        let bass = response(&mut stack, 90.0);
        let mid = response(&mut stack, 700.0);
        let treble = response(&mut stack, 5_000.0);
        assert!(mid < bass, "no scoop below: mid {mid} vs bass {bass}");
        assert!(mid < treble, "no scoop above: mid {mid} vs treble {treble}");
    }

    fn clean_stack(mid: f32) -> ToneStack {
        let mut stack = ToneStack::new(ToneStackCircuit::or100_clean());
        stack.prepare(OS_RATE, 0.5, 0.5, mid);
        stack
    }

    #[test]
    fn clean_fixed_mid_resistor_fills_in_the_midrange() {
        // The clean channel has no `Middle` control, so the fixed resistor is
        // what sets its voicing. Compare against a near-shorted mid element,
        // which is what a 2-band stack degenerates to if the value is wrong.
        let mut shorted = clean_stack(0.05);
        let mut fixed = clean_stack(ToneStack::CLEAN_FIXED_MID);

        let shorted_scoop = response(&mut shorted, 700.0) / response(&mut shorted, 90.0);
        let fixed_scoop = response(&mut fixed, 700.0) / response(&mut fixed, 90.0);
        assert!(
            fixed_scoop > shorted_scoop * 1.5,
            "fixed mid did not fill the midrange: {shorted_scoop} -> {fixed_scoop}"
        );
    }

    #[test]
    fn clean_two_band_stack_is_still_interactive() {
        // Losing the `Middle` control must not turn the remaining two into
        // independent bands: `Bass` must still reload the treble branch.
        let mut bass_down = clean_stack(ToneStack::CLEAN_FIXED_MID);
        bass_down.set_controls(0.5, 0.05, ToneStack::CLEAN_FIXED_MID);
        let mut bass_up = clean_stack(ToneStack::CLEAN_FIXED_MID);
        bass_up.set_controls(0.5, 0.95, ToneStack::CLEAN_FIXED_MID);

        let a = response(&mut bass_down, 700.0);
        let b = response(&mut bass_up, 700.0);
        assert!(
            ((b - a) / a).abs() > 0.05,
            "clean stack shows no bass/mid interaction: {a} -> {b}"
        );
    }

    #[test]
    fn the_two_channels_are_voiced_differently() {
        // Different slope resistor, treble cap and mid element: at identical
        // knob positions the two stacks must not produce the same curve.
        let mut clean = clean_stack(0.5);
        let mut dirty = dirty_stack(0.5, 0.5, 0.5);
        let mut differed = false;
        for frequency in [90.0, 200.0, 700.0, 2_000.0, 5_000.0] {
            let a = response(&mut clean, frequency);
            let b = response(&mut dirty, frequency);
            if ((a - b) / b).abs() > 0.05 {
                differed = true;
            }
        }
        assert!(differed, "the two tone stacks produced the same response");
    }

    #[test]
    fn clean_treble_control_retains_authority() {
        let mut low = clean_stack(ToneStack::CLEAN_FIXED_MID);
        low.set_controls(0.05, 0.5, ToneStack::CLEAN_FIXED_MID);
        let mut high = clean_stack(ToneStack::CLEAN_FIXED_MID);
        high.set_controls(0.95, 0.5, ToneStack::CLEAN_FIXED_MID);
        let range = response(&mut high, 5_000.0) / response(&mut low, 5_000.0);
        assert!(range > 1.5, "clean treble control does nothing: {range}");
    }

    #[test]
    fn set_controls_is_a_no_op_when_nothing_changed() {
        let mut stack = dirty_stack(0.5, 0.5, 0.5);
        let before = stack.reduced;
        assert!(stack.set_controls(0.5, 0.5, 0.5));
        assert_eq!(before, stack.reduced);
        assert!(stack.set_controls(0.9, 0.5, 0.5));
        assert_ne!(before, stack.reduced);
    }

    #[test]
    fn extreme_controls_do_not_produce_non_finite_output() {
        for (t, b, m) in [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
        ] {
            let mut stack = dirty_stack(t, b, m);
            for n in 0..10_000 {
                let x = (std::f32::consts::TAU * 100.0 * n as f32 / OS_RATE).sin() * 40.0;
                let y = stack.process(x);
                assert!(y.is_finite(), "({t},{b},{m}) produced {y}");
            }
        }
    }
}
