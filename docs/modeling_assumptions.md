# Modeling Assumptions

This project studies a protection-driven overvoltage cascade mechanism. The
models are academic mechanism replicas and screening benchmarks, not forensic
network equivalents.

## Protected-Side Voltage

The relay-relevant voltage may be a collector-side, transformer-side, or plant
evacuation voltage rather than the upstream transmission bus voltage. PA-DVSA
therefore evaluates protected outputs explicitly. If the protected node exists
in ANDES, the screen uses that bus voltage directly. If it is hidden, the screen
uses a fixed-tap reconstruction with a bounded error envelope.

## Fixed Taps And Hidden Collector Measurements

A hidden protected voltage is represented as a conservative envelope around

\[
z_i = V_{\mathrm{upstream}} / n_i + \epsilon_i,
\]

where \(n_i\) is the tap ratio and \(\epsilon_i\) captures reconstruction error.
If the lower bound on the margin to pickup is not positive, the asset is marked
data-limited rather than certified safe.

## Reactive Absorption Loss

Many overvoltage cascades are driven by removal of inductive absorption. A plant
or load trip can remove MW and MVAr absorption at the same time. PA-DVSA keeps
the signed physical injection internally, then exposes the voltage-raising
component used by the screen.

## Fixed-Power-Factor RES Ramps

For fixed-power-factor operation, an active-power change also changes reactive
absorption. The candidate event library includes fixed-PF ramps because they can
consume protected-voltage margin even when the MW schedule change looks modest.
Voltage-mode IBR operation is modeled as an ablation that preserves or restores
local reactive support.

## Relay Timing

The screen evaluates finite-window pickup and dwell quantities over the
relay-relevant time window. The resolvent proxy is a ranking speedup only when a
channel passes the monotone-response check; otherwise direct finite-window
sampling is used.

## ACTIVSg2000 Academic Replica

The ACTIVSg2000 case is a modified synthetic benchmark. The replica embeds
report-aligned mechanism components: operator voltage-control actions,
collector-side overvoltage relays, named plant/collector trips, generator
protection, UFLS/UVLS-style defense actions, AC separation, HVDC blocking, and
island viability checks. It does not claim geographic equivalence to Iberia.
