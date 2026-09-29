# Structural probe of the frozen E42 free encoders: does FE on the raw state
# lose the length of the positional vector (LayerNorm after a bias-negligible
# first layer) and drown the block angle? Criteria fixed before the first run:
# the hypothesis is refuted if, on raw input, median radial/orth >= 0.5 or
# median angle/blockx >= 0.5.
import sys, glob, re, numpy as np, torch
sys.path.insert(0, "E43_external_target/code")
import e43_lib as L

eps = L.collect_gym_data(n_ep=20, seed=42)
X = torch.from_numpy(np.concatenate([e["s"] for e in eps]).astype(np.float32))
print("states", tuple(X.shape))
sc5 = torch.tensor([1/512, 1/512, 1/512, 1/512, 1/(2*np.pi)])

def probe(fe, pre, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = lambda x: fe(pre(x))
    with torch.no_grad():
        y = f(X)
        p = X[:, :4]
        c = 0.95
        Xr = X.clone(); Xr[:, :4] = c * p
        dr = (f(Xr) - y).norm(dim=1)
        r = torch.randn(p.shape, generator=g)
        r = r - (r * p).sum(1, keepdim=True) / (p * p).sum(1, keepdim=True) * p
        r = r / r.norm(dim=1, keepdim=True) * ((1 - c) * p.norm(dim=1, keepdim=True))
        Xo = X.clone(); Xo[:, :4] = p + r
        do = (f(Xo) - y).norm(dim=1)
        Xa = X.clone(); Xa[:, 4] = X[:, 4] + 0.1 * 2 * np.pi
        Xb = X.clone(); Xb[:, 2] = X[:, 2] + 0.1 * 512
        da = (f(Xa) - y).norm(dim=1); db = (f(Xb) - y).norm(dim=1)
        W, b = fe.net[0].weight, fe.net[0].bias
        bshare = b.norm() / (pre(X) @ W.T).norm(dim=1)
    return [(dr.median() / do.median()).item(),
            (da.median() / db.median()).item(),
            bshare.median().item()]

rows = {}
for f in sorted(glob.glob("E42_candidate_sweep/checkpoints/*_enc_at_freeze.pt")):
    fe = L.FE(); fe.load_state_dict(torch.load(f, map_location="cpu")); fe.eval()
    key = re.search(r"/(s\d+)_", f).group(1)
    rows.setdefault(key, []).append(probe(fe, lambda x: x) + probe(fe, lambda x: x * sc5))

names = ["raw radial/orth", "raw angle/blockx", "raw bias share",
         "scaled radial/orth", "scaled angle/blockx", "scaled bias share"]
for key, v in sorted(rows.items()):
    a = np.array(v)
    print(key, "n", len(a))
    for k, nm in enumerate(names):
        print(f"  {nm:20s} median {np.median(a[:, k]):.4f} "
              f"min {a[:, k].min():.4f} max {a[:, k].max():.4f}")
