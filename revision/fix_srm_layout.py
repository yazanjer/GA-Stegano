#!/usr/bin/env python
"""Put SRM feature files written before the srm_full key-sorting fix into the canonical (sorted-name) layout.

sealwatch 2025.9 fills the last submodels of its feature dict from a set, so their order
changed with PYTHONHASHSEED from one features_v2.py invocation to the next (each invocation
forks its workers, so the order is constant within one file).  For every file this script
recomputes three rows, finds the block permutation that reproduces them exactly, and
writes the columns in sorted-name order.  After the fix, cover features computed on three
different machines are identical, and randomly chosen stego rows match a fresh computation.
"""

import sys, json, numpy as np, multiprocessing as mp, os, shutil
sys.path.insert(0,'/workspace/GA-Stegano/src'); sys.path.insert(0,'/workspace/GA-Stegano/revision')
from amdt.steganalysis.srm_full import install
from common import cover_sets, load
from pathlib import Path
RUN=Path('/workspace/runs/stego'); RAW=RUN/'features_raw'; OUT=RUN/'features'
ids=json.load(open(RAW/'covers.json')); covers=cover_sets()['stego'][:1000]
assert [p.stem for p in covers]==ids
ROWS=[0,1,997]
def feats(img):
    install(); import sealwatch.srm as s
    f=s.extract(img); return {k: np.asarray(v,dtype=np.float64).ravel().astype(np.float32) for k,v in f.items()}
def img(name, r):
    if name.startswith('cover'): return load(covers[r])
    from PIL import Image; return np.array(Image.open(RUN/'stego'/name/f'{ids[r]}.png'))
def job(fn):
    name=fn[:-4]
    X=np.load(RAW/fn, mmap_mode='r')
    D=[feats(img(name,r)) for r in ROWS]
    K=sorted(D[0]); n={k:D[0][k].size for k in K}
    pos=0; used=set(); src={}
    while pos<X.shape[1]:
        c=[k for k in K if k not in used and pos+n[k]<=X.shape[1] and all(np.array_equal(X[r,pos:pos+n[k]],D[j][k]) for j,r in enumerate(ROWS))]
        if not c: return fn, 'FAIL at %d'%pos
        k=c[0]; used.add(k); src[k]=pos; pos+=n[k]
    idx=np.concatenate([np.arange(src[k],src[k]+n[k]) for k in K])
    Y=np.ascontiguousarray(np.asarray(X)[:,idx]); np.save(OUT/fn, Y)
    ident=bool((idx==np.arange(idx.size)).all())
    return fn, 'ok identity=%s'%ident
if __name__=='__main__':
    OUT.mkdir(exist_ok=True); shutil.copy(RAW/'covers.json', OUT/'covers.json')
    fs=sorted(f for f in os.listdir(RAW) if f.endswith('.npy'))
    with mp.get_context('fork').Pool(12) as p:
        for r in p.imap_unordered(job, fs): print(*r, flush=True)
    print('FIXDONE')
