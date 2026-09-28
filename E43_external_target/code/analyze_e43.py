#!/usr/bin/env python3
"""E43 analysis, exactly as pre-registered in EXPERIMENTS.md, E43.

Read-only: reads E43 and E42 result files, writes nothing. Prints the three
pre-registered correlations, the jackknife, the descriptive quantities R,
ordering agreement and noise floor F, and the declared outcome they select.
If any of the 30 primary cells is missing it prints what is missing and gives
no verdict.
"""
import json
import math

ROOT = "/mnt/d/coordinate-stability"
E43 = json.load(open(f"{ROOT}/E43_external_target/results/sweep.json"))
E42 = json.load(open(f"{ROOT}/E42_candidate_sweep/results/sweep.json"))["cells"]
C = E43["cells"]

ALPHA = 0.05 / 3
PREDICTORS = [
    ("log_persistence", lambda c: math.log(c["persistence"]), "none"),
    ("r2_readout",      lambda c: c["r2_readout"],            "negative"),
    ("eff_rank",        lambda c: c["eff_rank"],              "negative"),
]


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    return pearson(rank(x), rank(y))


def fisher_p(r, n):
    z = math.atanh(r) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2))


def sd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


def main():
    keys = [f"s42_i{k}" for k in range(1, 31)]
    missing = [k for k in keys if k not in C]
    if missing:
        print(f"primary cells missing: {missing}\nno verdict")
        return

    assert all(C[k].get("encoder_matches_e42") for k in keys), "encoder check absent"
    y = [math.log(C[k]["final_vp"]) for k in keys]
    y42 = [math.log(E42[k]["final_vp"]) for k in keys]
    n = len(keys)

    print(f"E43 seed 42, n={n}, target {E43.get('target')}, "
          f"prereg {E43.get('prereg_commit')}")
    fv = [C[k]["final_vp"] for k in keys]
    print(f"final_vp spread {max(fv)/min(fv):.4f}x  sd(log) {sd(y):.4f}")
    print(f"E42 final_vp on the same encoders: sd(log) {sd(y42):.4f}")

    print(f"\n{'predictor':<16} {'pearson':>8} {'p_fisher':>9} {'hit':>4} "
          f"{'predicted':>10} {'spearman':>9}")
    res = {}
    for name, f, pred in PREDICTORS:
        x = [f(C[k]) for k in keys]
        r = pearson(x, y)
        p = fisher_p(r, n)
        hit = p < ALPHA
        res[name] = (r, p, hit, pred)
        print(f"{name:<16} {r:>+8.4f} {p:>9.5f} {('yes' if hit else 'no'):>4} "
              f"{pred:>10} {spearman(x, y):>+9.4f}")

    print("\njackknife, every predictor")
    for name, f, _pred in PREDICTORS:
        x = [f(C[k]) for k in keys]
        loo = []
        for i in range(n):
            xs = x[:i] + x[i + 1:]
            ys = y[:i] + y[i + 1:]
            loo.append((pearson(xs, ys), keys[i]))
        lo, hi = min(loo), max(loo)
        print(f"  {name:<16} full {res[name][0]:+.4f}  range [{lo[0]:+.4f} ({lo[1]}), "
              f"{hi[0]:+.4f} ({hi[1]})]")

    R = sd(y) / sd(y42)
    agree = spearman(y, y42)
    print(f"\nR = sd(log E43) / sd(log E42) = {R:.4f}")
    print(f"ordering agreement, Spearman E43 vs E42 final_vp: {agree:+.4f}")

    floor_keys = ["s42_i1"] + [f"s42_i1_h{j}" for j in range(1, 6)]
    fmiss = [k for k in floor_keys if k not in C]
    F = None
    if fmiss:
        print(f"noise floor: missing {fmiss}")
    else:
        fl = [math.log(C[k]["final_vp"]) for k in floor_keys]
        F = sd(fl) / sd(y)
        print(f"noise floor: sd(log) over 6 heads on encoder 1 {sd(fl):.4f}, F = {F:.4f}")

    k123 = [f"s123_i{k}" for k in range(1, 11)]
    if all(k in C for k in k123):
        a = [math.log(C[k]["final_vp"]) for k in k123]
        b = [math.log(E42[k]["final_vp"]) for k in k123]
        print(f"seed 123: R = {sd(a)/sd(b):.4f}  (reproduction only)")
    else:
        print(f"seed 123: {sum(k in C for k in k123)}/10 cells present")

    print("\nDECLARED OUTCOMES")
    if F is None:
        print("  noise floor not measured: correlations are reported, "
              "interpretation waits for the floor cells")
    elif F >= 0.5:
        print(f"  F = {F:.4f} >= 0.5: correlations reported, NOT interpreted")
    p1_hit = res["log_persistence"][2]
    if R < 0.5 and not p1_hit:
        print("  COLLAPSE: the E42 spread is mainly a property of the "
              "self-referential target")
    if R >= 0.5:
        print("  SPREAD PERSISTS on a target the encoders do not define")
        leads = []
        for name in ("r2_readout", "eff_rank"):
            r, p, hit, pred = res[name]
            if hit and (r < 0) == (pred == "negative"):
                leads.append(name)
            elif hit:
                print(f"  {name}: hit against the prediction (r {r:+.4f})")
        print(f"  leads: {leads if leads else 'none; the difference is real and none of the three orders it'}")
    if p1_hit:
        print("  PREDICTOR 1 HIT: persistence carries information beyond the "
              "self-referential scale")
    if R < 0.5 and p1_hit:
        print("  R < 0.5 with predictor 1 a hit: no collapse verdict, see above")


if __name__ == "__main__":
    main()
