//! Long-tailed-pair phase inverter, push-pull EL34 output stage, and the
//! dynamic `B+` sag tracker.
//!
//! Specification section 2.C and `CLAUDE.md` §2.

use super::denormal::sanitize;
use super::filters::EnvelopeFollower;
use super::triode::{StageCircuit, Triode};

/// Reference plate voltage the drive normalization is calibrated against, in
/// volts. Equal to the 100 W mode's nominal rail so that mode needs no
/// correction factor.
pub const REFERENCE_B_PLUS: f32 = 480.0;

/// Grid drive, in volts, that takes an EL34 from idle to full conduction at
/// [`REFERENCE_B_PLUS`]. An EL34 biased near -38 V reaches the top of its
/// transfer curve on roughly this much swing.
const FULL_CONDUCTION_VOLTS: f32 = 34.0;

/// Class-AB idle point, in normalized grid-drive units.
///
/// Specification section 2.C states the push-pull transfer as
/// `tanh(V+/Vsag) - tanh(V-/Vsag)`, which is the class-A case. Real EL34s idle
/// near cutoff, so both tubes sit at a negative offset and neither conducts
/// over the first fraction of a volt — the crossover notch the specification
/// also calls for. Setting this constant to `0.0` recovers the specification's
/// bare formula exactly.
const CLASS_AB_BIAS: f32 = -1.05;

/// Fraction of the nominal rail the supply drops at sustained full output.
/// A choke-filtered EL34 amplifier's `B+` sags by 12..18 % under load.
const SAG_DEPTH: f32 = 0.16;

/// Attack time of the sag envelope, in ms (specification section 2.C).
const SAG_ATTACK_MS: f32 = 8.0;
/// Release time of the sag envelope, in ms (specification section 2.C).
const SAG_RELEASE_MS: f32 = 120.0;

/// Output power switching matrix from specification section 2.C.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerMode {
    /// 4 tubes, 480 V rail.
    Watt100,
    /// 4 tubes, rail stepped down to 340 V.
    Watt70,
    /// 2 tubes (inner pair disabled), 480 V rail.
    Watt50,
    /// 2 tubes, rail stepped down to 340 V.
    Watt30,
}

impl PowerMode {
    /// Number of EL34s conducting in this mode.
    pub const fn active_tubes(self) -> f32 {
        match self {
            PowerMode::Watt100 | PowerMode::Watt70 => 4.0,
            PowerMode::Watt50 | PowerMode::Watt30 => 2.0,
        }
    }

    /// Nominal `B+` rail voltage for this mode.
    pub const fn nominal_b_plus(self) -> f32 {
        match self {
            PowerMode::Watt100 | PowerMode::Watt50 => 480.0,
            PowerMode::Watt70 | PowerMode::Watt30 => 340.0,
        }
    }

    /// Nameplate output power in watts.
    pub const fn rated_watts(self) -> f32 {
        match self {
            PowerMode::Watt100 => 100.0,
            PowerMode::Watt70 => 70.0,
            PowerMode::Watt50 => 50.0,
            PowerMode::Watt30 => 30.0,
        }
    }

    /// Output voltage scaling relative to the 100 W mode.
    ///
    /// Power goes as the square of voltage, so a mode rated at `W` watts
    /// delivers `sqrt(W / 100)` of the 100 W mode's voltage swing into the same
    /// load. This is what makes the half-power modes quieter *as well as*
    /// earlier-breaking.
    pub fn output_scale(self) -> f32 {
        (self.rated_watts() / 100.0).sqrt()
    }
}

/// Long-tailed-pair phase inverter built from two real 12AX7 triode models.
///
/// A shortcut here would be a pair of gains and a `tanh`, but the LTP's own
/// triodes clip well before the EL34s do on a cranked amp, and `CLAUDE.md` §2
/// rules out naive waveshapers for 12AX7s. Each leg therefore runs a full
/// [`Triode`].
///
/// The shared tail resistor is represented by [`Self::TAIL_IMBALANCE`] rather
/// than solved explicitly: in a real LTP the inverting leg is driven through
/// the common cathode impedance and comes out a few percent quieter than the
/// driven leg. That residual imbalance is precisely what gives a cranked
/// push-pull amplifier its second-harmonic content.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct PhaseInverter {
    positive_leg: Triode,
    negative_leg: Triode,
    drive: f32,
}

impl PhaseInverter {
    /// Ratio of the inverting leg's drive to the non-inverting leg's.
    ///
    /// A Marshall-style LTP with a 10 kΩ tail typically lands within 4..7 % of
    /// balance.
    pub const TAIL_IMBALANCE: f32 = 0.945;

    /// Voltage gain from the inverter's input to each grid before the triode
    /// models are applied. The LTP presents a modest gain; most of the swing it
    /// needs already comes from the preamp.
    const INPUT_GAIN: f32 = 0.6;

    /// LTP circuit values: 82 kΩ plate loads off a 400 V rail with small
    /// unbypassed 470 Ω cathode resistors. The high cathode-network corner
    /// models the *unbypassed* cathodes an LTP requires, giving each leg strong
    /// local degeneration.
    fn circuit() -> StageCircuit {
        StageCircuit {
            plate_resistor: 82_000.0,
            supply_voltage: 400.0,
            cathode_resistor: 470.0,
            cathode_bypass_hz: 40_000.0,
            coupling_cap: 22.0e-9,
            grid_leak: 1.0e6,
            grid_conduction_resistance: 2_200.0,
            miller_cutoff_hz: 60_000.0,
        }
    }

    /// Builds an unprepared inverter.
    pub fn new() -> Self {
        Self {
            positive_leg: Triode::new(Self::circuit()),
            negative_leg: Triode::new(Self::circuit()),
            drive: Self::INPUT_GAIN,
        }
    }

    /// Builds both triode tables and configures the filters.
    pub fn prepare(&mut self, sample_rate: f32) {
        self.positive_leg.prepare(sample_rate);
        self.negative_leg.prepare(sample_rate);
    }

    /// Returns both legs to their quiescent state.
    pub fn reset(&mut self) {
        self.positive_leg.reset();
        self.negative_leg.reset();
    }

    /// Produces the differential drive pair `[V+, V-]` for one sample.
    #[inline(always)]
    pub fn process(&mut self, input: f32) -> [f32; 2] {
        let driven = input * self.drive;
        // The triode model already inverts, so the "positive" leg is the one
        // fed directly and the "negative" leg is fed the inverted signal,
        // attenuated by the tail imbalance.
        let positive = self.positive_leg.process(driven);
        let negative = self.negative_leg.process(-driven * Self::TAIL_IMBALANCE);
        [positive, negative]
    }
}

impl Default for PhaseInverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Push-pull EL34 output stage with a dynamic `B+` rail.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct PowerAmp {
    mode: PowerMode,
    /// Dual-time-constant follower tracking total power-stage current draw.
    sag_envelope: EnvelopeFollower,
    /// Fraction of the nominal rail currently available, driven by the
    /// power/standby switch and the warm-up ramp in
    /// [`crate::dsp::engine`]. `0.0` on standby, `1.0` once warm.
    rail_availability: f32,
    /// Most recent sag-adjusted rail voltage, exposed for the jewel lamp.
    current_b_plus: f32,
    /// Total conduction of both tubes at idle, `2 * tanh(CLASS_AB_BIAS)`,
    /// precomputed as the reference the current draw is measured against.
    idle_conduction: f32,
}

impl PowerAmp {
    /// Builds an unprepared power stage in 100 W mode.
    pub fn new() -> Self {
        Self {
            mode: PowerMode::Watt100,
            sag_envelope: EnvelopeFollower::default(),
            rail_availability: 1.0,
            current_b_plus: PowerMode::Watt100.nominal_b_plus(),
            idle_conduction: 2.0 * CLASS_AB_BIAS.tanh(),
        }
    }

    /// Configures the sag envelope for `sample_rate` and clears the state.
    pub fn prepare(&mut self, sample_rate: f32) {
        self.sag_envelope
            .prepare(SAG_ATTACK_MS, SAG_RELEASE_MS, sample_rate);
        self.idle_conduction = 2.0 * CLASS_AB_BIAS.tanh();
        self.reset();
    }

    /// Clears the sag envelope back to a fully charged rail.
    pub fn reset(&mut self) {
        self.sag_envelope.reset();
        self.current_b_plus = self.mode.nominal_b_plus() * self.rail_availability;
    }

    /// Selects the wattage mode. Cheap enough to call every block.
    pub fn set_mode(&mut self, mode: PowerMode) {
        self.mode = mode;
    }

    /// Currently selected wattage mode.
    pub fn mode(&self) -> PowerMode {
        self.mode
    }

    /// Sets how much of the nominal rail the power supply is delivering, in
    /// `0.0..=1.0`. The engine drives this from the power/standby switch and
    /// the tube warm-up ramp.
    pub fn set_rail_availability(&mut self, availability: f32) {
        self.rail_availability = availability.clamp(0.0, 1.0);
    }

    /// Instantaneous `B+` rail voltage after sag, in volts.
    ///
    /// Read by the GUI's jewel lamp through an atomic in
    /// [`crate::dsp::engine`]; never read from the GUI thread directly.
    pub fn b_plus(&self) -> f32 {
        self.current_b_plus
    }

    /// Processes one differential drive pair into a single-ended output.
    ///
    /// Implements specification section 2.C:
    ///
    /// ```text
    /// Vsag(t) = Vnominal - dV * Env(|Iout|)
    /// Vout    = tanh(V+/Vsag) - tanh(V-/Vsag)
    /// ```
    ///
    /// with the class-AB idle offset described at [`CLASS_AB_BIAS`].
    #[inline(always)]
    pub fn process(&mut self, drive: [f32; 2]) -> f32 {
        let nominal = self.mode.nominal_b_plus() * self.rail_availability;
        let tubes = self.mode.active_tubes();

        // Sag depth scales with how many tubes are drawing from the rail: the
        // 2-tube modes load the same supply half as hard.
        let sag_drop = SAG_DEPTH * nominal * (tubes * 0.25) * self.sag_envelope.value();
        let rail = (nominal - sag_drop).max(0.0);
        self.current_b_plus = rail;

        // Grid volts needed for full conduction, tracking the sagging rail.
        // The floor keeps the division safe when the amp is on standby.
        let scale = (FULL_CONDUCTION_VOLTS * (rail / REFERENCE_B_PLUS)).max(1.0e-3);
        let normalized_positive = drive[0] / scale;
        let normalized_negative = drive[1] / scale;

        let upper = (normalized_positive + CLASS_AB_BIAS).tanh();
        let lower = (normalized_negative + CLASS_AB_BIAS).tanh();

        // Total cathode current above idle drives the sag envelope. Both tubes
        // conducting hard at once is what actually loads the supply, so this
        // uses the sum, not the difference that forms the output.
        let draw = ((upper + lower) - self.idle_conduction).abs();
        self.sag_envelope.process(draw);

        // Rail collapse also scales the achievable output swing, which is why
        // a sagging amp compresses rather than merely distorting.
        let swing = rail / REFERENCE_B_PLUS;
        sanitize((upper - lower) * swing * self.mode.output_scale())
    }
}

impl Default for PowerAmp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OS_RATE: f32 = 384_000.0;

    fn prepared_power_amp(mode: PowerMode) -> PowerAmp {
        let mut amp = PowerAmp::new();
        amp.prepare(OS_RATE);
        amp.set_mode(mode);
        amp.reset();
        amp
    }

    #[test]
    fn power_mode_matrix_matches_the_specification() {
        assert_eq!(PowerMode::Watt100.active_tubes(), 4.0);
        assert_eq!(PowerMode::Watt100.nominal_b_plus(), 480.0);
        assert_eq!(PowerMode::Watt70.active_tubes(), 4.0);
        assert_eq!(PowerMode::Watt70.nominal_b_plus(), 340.0);
        assert_eq!(PowerMode::Watt50.active_tubes(), 2.0);
        assert_eq!(PowerMode::Watt50.nominal_b_plus(), 480.0);
        assert_eq!(PowerMode::Watt30.active_tubes(), 2.0);
        assert_eq!(PowerMode::Watt30.nominal_b_plus(), 340.0);
    }

    #[test]
    fn output_scale_falls_monotonically_with_rating() {
        let scales = [
            PowerMode::Watt100.output_scale(),
            PowerMode::Watt70.output_scale(),
            PowerMode::Watt50.output_scale(),
            PowerMode::Watt30.output_scale(),
        ];
        for pair in scales.windows(2) {
            assert!(pair[0] > pair[1], "scales not monotonic: {scales:?}");
        }
        assert!((scales[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn idle_produces_no_output() {
        let mut amp = prepared_power_amp(PowerMode::Watt100);
        for _ in 0..1_000 {
            let y = amp.process([0.0, 0.0]);
            assert!(y.abs() < 1.0e-6, "idle output {y}");
        }
    }

    #[test]
    fn push_pull_transfer_is_odd_symmetric() {
        let mut amp = prepared_power_amp(PowerMode::Watt100);
        for volts in [1.0f32, 5.0, 15.0, 40.0] {
            let positive = amp.process([volts, -volts]);
            let negative = amp.process([-volts, volts]);
            assert!(
                (positive + negative).abs() < 1.0e-4,
                "asymmetry at {volts} V: {positive} vs {negative}"
            );
        }
    }

    #[test]
    fn crossover_region_has_reduced_gain() {
        let mut amp = prepared_power_amp(PowerMode::Watt100);
        // Incremental gain just off zero must be lower than in the linear
        // region above the crossover notch.
        let small = amp.process([0.4, -0.4]).abs() / 0.4;
        let medium = (amp.process([9.0, -9.0]).abs() - amp.process([7.0, -7.0]).abs()) / 2.0;
        assert!(small < medium, "no crossover notch: {small} vs {medium}");
    }

    #[test]
    fn output_saturates_rather_than_growing_without_bound() {
        let mut amp = prepared_power_amp(PowerMode::Watt100);
        // The class-AB offset means full conduction needs rather more than the
        // nominal 34 V; by 120 V both tubes are hard against their limits.
        let moderate = amp.process([120.0, -120.0]).abs();
        let extreme = amp.process([100_000.0, -100_000.0]).abs();
        assert!(extreme.is_finite());
        assert!(
            extreme < moderate * 1.05,
            "did not saturate: {moderate} -> {extreme}"
        );
        // `tanh(+1) - tanh(-1)` bounds the stage at exactly 2.0.
        assert!(extreme <= 2.0);
    }

    #[test]
    fn sag_pulls_the_rail_down_and_recovers() {
        let mut amp = prepared_power_amp(PowerMode::Watt100);
        assert!((amp.b_plus() - 480.0).abs() < 1.0);

        // 50 ms of full-tilt drive: well past the 8 ms attack.
        for n in 0..(0.05 * OS_RATE) as usize {
            let phase = std::f32::consts::TAU * 110.0 * n as f32 / OS_RATE;
            let v = 60.0 * phase.sin();
            amp.process([v, -v]);
        }
        let sagged = amp.b_plus();
        assert!(sagged < 460.0, "rail barely moved: {sagged} V");
        assert!(sagged > 350.0, "rail collapsed too far: {sagged} V");

        // 600 ms of silence: five release time constants.
        for _ in 0..(0.6 * OS_RATE) as usize {
            amp.process([0.0, 0.0]);
        }
        assert!(
            amp.b_plus() > 475.0,
            "rail failed to recover: {} V",
            amp.b_plus()
        );
    }

    #[test]
    fn sag_attack_is_faster_than_release() {
        let mut amp = prepared_power_amp(PowerMode::Watt100);
        let drive_for = |amp: &mut PowerAmp, seconds: f32| {
            for n in 0..(seconds * OS_RATE) as usize {
                let phase = std::f32::consts::TAU * 110.0 * n as f32 / OS_RATE;
                let v = 60.0 * phase.sin();
                amp.process([v, -v]);
            }
        };
        drive_for(&mut amp, 0.008);
        let after_attack = 480.0 - amp.b_plus();
        drive_for(&mut amp, 0.2);
        let settled = 480.0 - amp.b_plus();
        // One attack time constant should already have reached ~63 % of the
        // final drop.
        assert!(after_attack > settled * 0.4, "attack too slow");

        for _ in 0..(0.008 * OS_RATE) as usize {
            amp.process([0.0, 0.0]);
        }
        let after_release = 480.0 - amp.b_plus();
        // ...but 8 ms of release must barely move it, at a 120 ms constant.
        assert!(
            after_release > settled * 0.85,
            "release far too fast: {settled} -> {after_release}"
        );
    }

    #[test]
    fn two_tube_modes_sag_less_than_four_tube_modes() {
        let measure = |mode: PowerMode| {
            let mut amp = prepared_power_amp(mode);
            for n in 0..(0.2 * OS_RATE) as usize {
                let phase = std::f32::consts::TAU * 110.0 * n as f32 / OS_RATE;
                let v = 60.0 * phase.sin();
                amp.process([v, -v]);
            }
            (mode.nominal_b_plus() - amp.b_plus()) / mode.nominal_b_plus()
        };
        assert!(measure(PowerMode::Watt100) > measure(PowerMode::Watt50));
    }

    #[test]
    fn lower_rail_modes_clip_earlier() {
        // Both probes sit above the crossover notch, so this measures
        // saturation rather than how far each mode has climbed out of its
        // crossover region. Doubling the drive gives less than double the
        // output, and the 340 V rail gives less still.
        let expansion = |mode: PowerMode| {
            let mut amp = prepared_power_amp(mode);
            let small = amp.process([30.0, -30.0]).abs();
            let large = amp.process([60.0, -60.0]).abs();
            large / small
        };
        let full_rail = expansion(PowerMode::Watt100);
        let stepped_down = expansion(PowerMode::Watt70);
        assert!(
            stepped_down < full_rail,
            "the 340 V rail should compress sooner: {stepped_down} vs {full_rail}"
        );

        // Further up the curve, past the crossover region entirely, even the
        // 480 V rail must show unambiguous compression.
        let mut amp = prepared_power_amp(PowerMode::Watt100);
        let at_60 = amp.process([60.0, -60.0]).abs();
        let at_120 = amp.process([120.0, -120.0]).abs();
        assert!(
            at_120 < at_60 * 1.5,
            "the 480 V rail did not compress at all: {at_60} -> {at_120}"
        );
    }

    #[test]
    fn standby_silences_the_stage() {
        let mut amp = prepared_power_amp(PowerMode::Watt100);
        amp.set_rail_availability(0.0);
        for volts in [1.0f32, 20.0, 100.0] {
            let y = amp.process([volts, -volts]);
            assert!(y.abs() < 1.0e-6, "standby leaked {y} at {volts} V");
        }
        assert_eq!(amp.b_plus(), 0.0);
    }

    #[test]
    fn extreme_input_stays_finite() {
        let mut amp = prepared_power_amp(PowerMode::Watt30);
        for drive in [[1.0e9f32, -1.0e9], [f32::MAX, f32::MIN], [-1.0e9, 1.0e9]] {
            let y = amp.process(drive);
            assert!(y.is_finite(), "{drive:?} produced {y}");
        }
    }

    #[test]
    fn phase_inverter_produces_an_inverted_pair() {
        let mut inverter = PhaseInverter::new();
        inverter.prepare(OS_RATE);
        let mut positive_peak = 0.0f32;
        let mut negative_peak = 0.0f32;
        let total = (OS_RATE / 100.0) as usize * 4;
        for n in 0..total {
            // Small signal: both legs clip to the same rail under heavy drive,
            // which would hide the imbalance entirely.
            let x = 0.02 * (std::f32::consts::TAU * 100.0 * n as f32 / OS_RATE).sin();
            let [p, m] = inverter.process(x);
            if n > total / 2 {
                positive_peak = positive_peak.max(p.abs());
                negative_peak = negative_peak.max(m.abs());
            }
        }
        assert!(positive_peak > 0.0 && negative_peak > 0.0);
        // The tail imbalance must be present but small.
        let ratio = negative_peak / positive_peak;
        assert!(
            (0.85..=0.99).contains(&ratio),
            "leg balance ratio was {ratio}"
        );
    }

    #[test]
    fn phase_inverter_legs_are_out_of_phase() {
        let mut inverter = PhaseInverter::new();
        inverter.prepare(OS_RATE);
        // Settle the DC blockers first.
        for _ in 0..10_000 {
            inverter.process(0.0);
        }
        let mut correlation = 0.0f32;
        for n in 0..20_000 {
            let x = 0.02 * (std::f32::consts::TAU * 100.0 * n as f32 / OS_RATE).sin();
            let [p, m] = inverter.process(x);
            correlation += p * m;
        }
        assert!(correlation < 0.0, "legs are in phase: {correlation}");
    }

    #[test]
    fn phase_inverter_output_is_bounded() {
        let mut inverter = PhaseInverter::new();
        inverter.prepare(OS_RATE);
        for x in [500.0f32, -500.0, 1.0e9, f32::MIN] {
            let [p, m] = inverter.process(x);
            assert!(p.is_finite() && m.is_finite(), "{x} produced {p}/{m}");
        }
    }
}
