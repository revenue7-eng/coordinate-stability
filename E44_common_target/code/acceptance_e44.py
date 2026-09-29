# E44 acceptance, run before pre-registration. Reads no external loss of a
# free arm: free_raw runs stage 1 only; prescribed runs both stages, and its
# external target equals its own latent.
import sys, json, time
sys.path.insert(0, "E44_common_target/code")
import e44_lib as L
R = json.load(open("E28_dim_sweep_full/results/p2_dim_sweep_results.json"))["results"]
seed = 42
for arm, key, stage2 in (("prescribed", "prescribed_dim5_seed42", True),
                         ("free_raw", "free_dim5_seed42", False)):
    t0 = time.time()
    tl, vl = L.make_data(seed)
    r = L.run_arm(arm, tl, vl, seed, stage2=stage2)
    print(arm, "stage1 best", repr(r["s1_best"]), "E28", repr(R[key]),
          "bit-exact:", r["s1_best"] == R[key], "s1_seconds", r["s1_seconds"])
    if stage2:
        print(arm, "stage2 hist equals stage1 hist:", r["s2_hist"] == r["s1_hist"],
              "s2_seconds", r["s2_seconds"])
    print(arm, "cell seconds incl. data", round(time.time() - t0, 1))
