# E44 precondition: free_raw stage 1 at seed 42 a second time in the current
# environment, with the committed library, for comparison with the acceptance
# run (not with E28).
import sys, time
sys.path.insert(0, "E44_common_target/code")
import e44_lib as L
t0 = time.time()
tl, vl = L.make_data(42)
r = L.run_arm("free_raw", tl, vl, 42, stage2=False)
print("free_raw stage1 best rerun", repr(r["s1_best"]), "seconds", round(time.time() - t0, 1), flush=True)
