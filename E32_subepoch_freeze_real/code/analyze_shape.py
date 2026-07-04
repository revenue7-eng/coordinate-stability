import json, glob, numpy as np
import os
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results')
band = [0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]
bk = ['%.2f'%f for f in band]
seeds = {}
for p in sorted(glob.glob(OUT+'/seed_*.json')):
    d=json.load(open(p)); seeds[d['seed']]=d

# ---- per-seed monotonicity over band ----
mono_full=0; dips_total=0; per_seed=[]
for s,d in seeds.items():
    y=[d['sweep'][k] for k in bk]
    dips=sum(1 for i in range(1,len(y)) if y[i]<y[i-1])
    per_seed.append((s,dips,y))
    if dips==0: mono_full+=1
    dips_total+=dips

# ---- pooled linear vs step over band (min-max normalized per seed) ----
X=[]; Y=[]
for s,d in seeds.items():
    y=np.array([d['sweep'][k] for k in bk],float)
    yn=(y-y.min())/(y.max()-y.min()+1e-12)   # normalize band to [0,1]
    X+=list(band); Y+=list(yn)
X=np.array(X); Y=np.array(Y)

# linear fit
A=np.vstack([X,np.ones_like(X)]).T
coef,_,_,_=np.linalg.lstsq(A,Y,rcond=None)
pred=A@coef; ss_lin=float(((Y-pred)**2).sum())
ss_tot=float(((Y-Y.mean())**2).sum()); r2_lin=1-ss_lin/ss_tot

# best single-breakpoint step fit (two constants)
ss_step=np.inf; best_bp=None
for bp in range(1,len(band)):
    thr=band[bp]
    left=Y[X<thr]; right=Y[X>=thr]
    if len(left)==0 or len(right)==0: continue
    ss=((left-left.mean())**2).sum()+((right-right.mean())**2).sum()
    if ss<ss_step: ss_step=float(ss); best_bp=thr
ratio=ss_step/ss_lin

# ---- E30-style anchors (real data) ----
anch=[]
for s,d in seeds.items():
    f0=d['sweep']['0.00']; f1=d['sweep']['1.00']
    anch.append(f1/f0)
anch=np.array(anch)

# ---- verdict (from code) ----
if ratio>=1.3 and r2_lin>=0.75 and mono_full>=3:
    verdict='SLOPE'
elif ratio<=0.7:
    verdict='STEP'
else:
    verdict='INCONCLUSIVE'

print('='*58)
print('E32 SHAPE ANALYSIS (real gym-pusht, within epoch 1)')
print('='*58)
print(f'seeds: {sorted(seeds)}  band: {band[0]}–{band[-1]}')
print(f'\nMonotonicity over band:')
for s,dips,y in per_seed:
    print(f'  seed {s:>4}: dips={dips}  vals={[round(v,4) for v in y]}')
print(f'  fully-monotone seeds: {mono_full}/{len(seeds)}   total downward dips: {dips_total}')
print(f'\nLinear vs step (pooled, per-seed min-max normalized):')
print(f'  SS_linear = {ss_lin:.4f}   R2_linear = {r2_lin:.3f}')
print(f'  SS_step   = {ss_step:.4f}   (best breakpoint @ f={best_bp})')
print(f'  SS_step / SS_linear = {ratio:.2f}x   (>1 => linear better => slope)')
print(f'\nE30-style anchor freeze@1.0 / freeze@0.0 (real):')
print(f'  ratios per seed: {[round(a,1) for a in anch]}')
print(f'  mean cliff = {anch.mean():.1f}x')
print(f'\n>>> CODE VERDICT: {verdict}')
json.dump({'verdict':verdict,'ss_lin':ss_lin,'ss_step':ss_step,'ratio':ratio,
           'r2_lin':r2_lin,'mono_full':mono_full,'dips_total':dips_total,
           'anchor_cliff_mean':float(anch.mean()),'seeds':sorted(seeds)},
          open(OUT+'/shape_verdict.json','w'),indent=2)
