# E44 feasibility: how much of the t+2 -> t+3 move is predictable from true
# states and recorded actions, with no encoder involved. Criterion fixed
# before the first run: the agent target is usable if R2_dyn(agent) >= 0.5.
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "E43_external_target/code")
import e43_lib as L

torch.manual_seed(0)
rng = np.random.default_rng(0)
t0 = time.time()
eps = L.collect_gym_data(seed=42)
print("episodes", len(eps), "collect_s", round(time.time() - t0, 1))

idx = rng.permutation(len(eps))
n_tr = int(0.8 * len(eps))
def windows(sel):
    d = L.DS([eps[i] for i in sel])
    S = np.stack([w[0] for w in d.w]); A = np.stack([w[1] for w in d.w])
    return torch.from_numpy(S), torch.from_numpy(A)
Str, Atr = windows(idx[:n_tr]); Sva, Ava = windows(idx[n_tr:])
print("windows train", len(Str), "val", len(Sva))

sc = torch.tensor([1/512, 1/512, 1/512, 1/512, 1/(2*np.pi)])
def feats(S, A):
    return torch.cat([(S[:, :3] * sc).reshape(len(S), -1),
                      (A[:, :3] / 512).reshape(len(A), -1)], -1)
def target(S, name, k):
    sl = slice(0, 2) if name == "agent" else slice(2, 5)
    return (S[:, k] * sc)[:, sl]

Xtr, Xva = feats(Str, Atr), feats(Sva, Ava)
for name in ("agent", "block"):
    ytr, yva = target(Str, name, 3), target(Sva, name, 3)
    pers = ((target(Sva, name, 2) - yva) ** 2).mean().item()
    net = nn.Sequential(nn.Linear(Xtr.size(1), 128), nn.GELU(),
                        nn.Linear(128, 128), nn.GELU(),
                        nn.Linear(128, ytr.size(1)))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(60):
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), 256):
            j = perm[b:b + 256]
            loss = ((net(Xtr[j]) - ytr[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mse = ((net(Xva) - yva) ** 2).mean().item()
    var = yva.var(0).mean().item()
    print(f"{name}: var {var:.6f} persistence_mse {pers:.6f} "
          f"oracle_mse {mse:.6f} R2_dyn {1 - mse / pers:.4f} "
          f"R2_total {1 - mse / var:.4f}")
print("total_s", round(time.time() - t0, 1))
