# Runs acceptance_e44.py with one progress line per epoch. val_loss is wrapped
# from outside; e44_lib.py is not modified.
import sys, time, runpy
sys.path.insert(0, "E44_common_target/code")
import e44_lib as L
_orig = L.val_loss
_s = {"t": time.time(), "n": 0}
def val_loss(mdl, vl):
    v = _orig(mdl, vl)
    _s["n"] += 1
    print(f"    epoch-eval {_s['n']:3d}  {time.time() - _s['t']:7.1f}s  vp {v:.6f}", flush=True)
    return v
L.val_loss = val_loss
runpy.run_path("E44_common_target/code/acceptance_e44.py", run_name="__main__")
