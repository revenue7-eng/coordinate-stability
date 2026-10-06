# E49: Completeness of the JEPA latent and an end-to-end learned encoder on the E48 target (synthetic Push-T)

## What this tests
Whether the frozen JEPA latent is complete (probes to the true state) and how an encoder trained end to end on the target compares.

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

## Ф86: E49, the frozen JEPA latent is complete but nonlinearly laid out; the registered verdict is fixation, with two named limits
- STATUS (Н5): prescribed_std is the standardised state with no encoder. The result reads: a frozen JEPA latent of the full state is complete but laid out worse than the state itself. It does not test fixed coordinates of a learned representation.

- Gates all true: one pre-registration commit; JEPA stage 1 equals E45 bit for bit on every seed; on prescribed_std the linear probe R2 of the four positions is at least 0.999. [verified: E49_completeness/results/analysis.json]
- Probes of the frozen JEPA latent of free_scaled, median over 10 seeds. MLP R2: x_a 0.999, y_a 0.999, x_b 0.999, y_b 0.999, sin 0.999, cos 1.000. Linear R2: x_a 0.980, y_a 0.982, x_b 0.985, y_b 0.988, sin 0.558, cos 0.023. Completeness as registered: complete (minimum median MLP R2 0.999). [verified: analysis.json]
- free_e2e / prescribed_std on final: 2.622 [2.219, 3.100], class O2. Frozen JEPA (E48) / prescribed_std: 1.436 [1.256, 1.642], class O2. Frozen JEPA / free_e2e: 0.548 [0.461, 0.650]. [verified: analysis.json]
- Registered verdict: fixation: the JEPA latent is complete and even a target-trained learned encoder stays behind.
- LIMITS, named after the data: (1) free_e2e had 90 epochs in total against 90 + 90 for frozen JEPA, and kept the SIGReg term; that it loses even to frozen JEPA points to under-training, so it is a weak upper bound for a learned encoder. (2) The design does not separate fixed coordinates from a simple (affine, well conditioned) layout of the state: prescribed_std is both. The probes show the learned latent differs from it in layout, not in information.
- INTERPRETATION: the advantage is not a loss of information in the learned latent. Whether it is fixation as such (Г-k) or the simple geometry of the state in the latent (Г-l) is open; Ф39 (a random fixed linear basis matches prescribed) is consistent with either.
- NEXT: E50, a fixed but nonlinearly warped complete encoder, and the learned encoder end to end on an equal budget without SIGReg.

## Setup
From EXPERIMENTS.md, section E49.

Question. Ф85: a standardised fixed basis beats the frozen JEPA latent of free_scaled on a nonlinear target. Is that because the JEPA latent loses state information (Г-j, completeness), or because the coordinates are fixed (Г-k)?

Design, 200 episodes, 90 epochs, seeds 42, 123, 777, 1001 to 1007 (n = 10):
1. free_scaled stage 1 (JEPA on its own latent), identical to E45 and E48 stage 1. Its frozen latent is probed for the state at the same step, targets (x_a, y_a, x_b, y_b)/512, sin theta, cos theta: linear probe (least squares with bias) and MLP probe (5-64-64-6, 100 epochs, Adam 1e-3), fitted on the training split, R2 per target on the validation split. The same probes on prescribed_std are a sanity check.
2. free_e2e: FreeEncoderScaled trained end to end on the E48 target from the start (encoder not frozen, no JEPA stage), same loop and SIGReg weight.
prescribed_std and the frozen JEPA free_scaled on the same target are taken from E48 cells (same seeds, data and code path).
Code: E49_completeness/code/run_e49.py (imports e44_lib.py unchanged), analyze_e49.py.

Registered statistics. Completeness: per target the median over seeds of the MLP probe R2 on the JEPA latent; "incomplete" if the minimum over the six targets is below 0.95, "complete" if at least 0.99, otherwise unresolved. End to end: GMR free_e2e / prescribed_std on the final epoch, 95% t interval, classes O1 to O5 as E44.
Verdict: completeness (Г-j) if incomplete and free_e2e is O4 or O5; fixation (Г-k) if complete and free_e2e is O1, O2 or O3; otherwise mixed, reported as such.
Named, not deciding: linear probe R2; probes of the end-to-end encoder; GMR of frozen JEPA free_scaled over free_e2e.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; JEPA stage 1 equals E45 free_scaled s1_hist bit for bit on every seed; on prescribed_std the linear probe R2 of the four positions is at least 0.999 on every seed.

Cost: not estimated in advance; the monitor reports the measured rate.

Result: all gates true. Latent complete; free_e2e O2. Registered verdict: fixation, with two named limits. See Ф86.

## Files
- `code/__pycache__`
- `code/analyze_e49.py`
- `code/patch_e48res_e49prereg.py`
- `code/run_e49.py`
- `results/analysis.json`
- `results/cells`
- `results/worker_1.log`
- `results/worker_2.log`
- `results/worker_3.log`
- `results/worker_4.log`

## Facts
Ф86

## How to reproduce
```
OMP_NUM_THREADS=1 python3 -u E49_completeness/code/run_e49.py --prereg fcf51ea862d1fc10f9796829619749909481c726 --seeds 42 123 777 1001 1002 1003 1004 1005 1006 1007
python3 E49_completeness/code/analyze_e49.py
```

## Status
Summary table: latent complete, laid out worse (Н5).

- Ф86: STATUS (Н5): prescribed_std is the standardised state with no encoder. The result reads: a frozen JEPA latent of the full state is complete but laid out worse than the state itself. It does not test fixed coordinates of a learned representation.

<!-- marker: repo-state v3 applied -->
