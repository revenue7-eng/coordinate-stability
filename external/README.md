# External patches

Modifications to third-party repositories used by this programme. The clones
themselves live outside this repository; only the patches and the commit they
apply to are recorded here, so that a clone can be discarded and rebuilt.

## epiplexity-3aa12a1-trapezoid.patch

Upstream: `shikaiqiu/epiplexity` (Finzi et al., arXiv 2601.03220)
Applies to: commit `3aa12a1a2be6a413fe9eaa41374a6a46a4a0d4e100`
Local clone during this work: `/mnt/d/eca_gate/epiplexity`
Relevant hypothesis: Г17 (epiplexity vs identifiability)

`soph/train.py` computes `K_auc` with `np.trapz`, which NumPy 2.x removed.
The patch switches the single call to `np.trapezoid`, the drop-in replacement
with identical semantics. Nothing else is changed.

This matters beyond making the script run: `K_auc` is the quantity intended as
the calibration target against Finzi's published result, so it sits on the
critical path of any Г17 work.

Rebuild:

```
git clone https://github.com/shikaiqiu/epiplexity
cd epiplexity && git checkout 3aa12a1
git apply /path/to/external/epiplexity-3aa12a1-trapezoid.patch
```

Measured feasibility (local CPU, minimum config L=1, D=16, B=32): 3.06 s/it,
a full grid point not under 8.5 hours, so a 30-point sweep is not local work.
Scaling by batch size and model size was not measured.
