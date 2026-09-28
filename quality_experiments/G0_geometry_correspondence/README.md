# G0 geometry / correspondence audit

G0 is a read-only diagnostic of the FrozenMLP baseline's direct central/no-PSF T1/T2 exports. For every native group, it scores every reconstructed group under discrete orientation candidates and solves a Hungarian assignment. The resulting match is **diagnostic only**: official pairing remains native group `g` to reconstructed group `g`, and G0 never rewrites D1 or other metrics.

The optional signal adapter consumes D2-compatible observation-level signal artifacts and selects unique weight-0/HB1 rows only. If those artifacts are absent, G0-map still runs and signal status is `DEPENDENCY_PENDING`. Pose measurements are reported when `final_rigid_poses.json` contains the physical post-stack-initial and final axis-angle arrays; large drift is measured, not automatically labeled incorrect.
