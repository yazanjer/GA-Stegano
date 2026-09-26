
import sys, os, json, numpy as np, multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS","1")
sys.path.insert(0,'/workspace/GA-Stegano/revision')
from common import cover_sets, load
from PIL import Image
from amdt.steganalysis.targeted import rs_analysis, weighted_stego_estimate
from amdt.stego.codec import header_bit_count
ids=[p.stem for p in cover_sets()['stego'][:1000]]
R='/workspace/runs/stego/stego/'
def one(a):
    s,i=a
    x=load(cover_sets()['stego'][ids.index(i)]) if s=='cover' else np.array(Image.open(f'{R}{s}/{i}.png'))
    return s, rs_analysis(x)['estimated_rate'], weighted_stego_estimate(x)['estimated_rate']
sets=['cover','AMDT@0.1','AMDT-D@0.1','LSB@0.1','LSB-M@0.1','EA-LSB@0.1','FM-PSO-LSB@0.1','HILL-STC@0.1','GA-FT@0.1']
with mp.Pool(8) as pool:
    res=pool.map(one,[(s,i) for s in sets for i in ids[:300]])
import collections
acc=collections.defaultdict(list)
for s,r,w in res: acc[s].append((r,w))
with open('/workspace/runs/stego/rsws.csv','w') as f:
    f.write('set,rs_mean,ws_mean,n\n')
    for s,v in acc.items():
        v=np.array(v); f.write(f'{s},{v[:,0].mean():.4f},{v[:,1].mean():.4f},{len(v)}\n')
# header statistics on AMDT@0.1 (real random nonces)
H=header_bit_count(4)
M=np.array([np.array(Image.open(f'{R}AMDT@0.1/{i}.png')).reshape(-1)[::-1][:H]&1 for i in ids])
magic=np.mean([np.packbits(m[:8])[0]==0xA7 for m in M])
cm=np.mean([np.packbits(load(p).reshape(-1)[::-1][:8]&1)[0]==0xA7 for p in cover_sets()['stego'][:1000]])
mu=M.mean(0)
json.dump({'header_bits':int(H),'n':len(M),'bit_mean_min':float(mu.min()),'bit_mean_max':float(mu.max()),
           'magic_rate_stego':float(magic),'magic_rate_cover':float(cm)},open('/workspace/runs/stego/header_stats.json','w'))
print('done')
