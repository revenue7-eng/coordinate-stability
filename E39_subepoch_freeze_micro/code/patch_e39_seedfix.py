#!/usr/bin/env python3
"""Seed the action space in E39's copy of the training library.

collect_gym_data seeds its own Generator (rng) and uses it for env.reset, the
branch draw and the noise draw, but env.action_space.sample() draws from the
action space's own generator, which gym.make initialises from system entropy.
That branch fires on the first step of every episode and in ~30% of later steps,
so roughly a third of all recorded actions came from an unseeded source and the
seed argument did not determine the dataset.

Applies to e39_lib.py only. e32_lib.py is shared with E32 and E38, whose numbers
are already in the registries, and is left untouched with the defect recorded.

Usage: python patch_e39_seedfix.py
"""
DST = "/mnt/d/coordinate-stability/E39_subepoch_freeze_micro/code/e39_lib.py"

src = open(DST).read()

OLD = '    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)'
NEW = (OLD + "\n"
       "    env.action_space.seed(int(rng.integers(0, 100000)))")

assert src.count("env.action_space.seed") == 0, "seed fix already applied"
n = src.count(OLD)
assert n == 1, f"E39-4 action_space seed: expected 1 match, found {n}"
src = src.replace(OLD, NEW)
open(DST, "w").write(src)
print("ok E39-4 action_space seed")
print(f"written {DST}")
