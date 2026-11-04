import numpy as np,glob,json,sys
sys.path.insert(0,'/home/g/george-vasilache/tmp/claude-11568/-apps-unit-DoyaU-vasilache-apps-code-robot/ecfef5f0-0409-4746-b1d6-0fa2c30562a7/scratchpad')
from runaway import segs
def per_min(n):
  run=glob.glob(f'e{n}_*')[0]
  L=[np.load(f) for f in sorted(glob.glob(run+'/logdir/online_shared/experience/*.npz'))]
  W=np.concatenate([d['wheels'] for d in L]); A=np.concatenate([d['drive'] for d in L]); O=np.concatenate([d['orientation'] for d in L])
  v=W.mean(1)/3.3e-4  # phone units (mm/s)
  pos=np.cumsum(v)/65/1000  # m
  d=A.mean(1); m=np.zeros(len(d),bool)
  for sg in (d>0.7,d<-0.7):
    for a,b in segs(sg,65): m[a:b]=True
  k=min(len(d)//3900,15); r=[]
  for i in range(k):
    s=slice(i*3900,(i+1)*3900)
    r.append(dict(rng=float(pos[s].max()-pos[s].min()), run=float(100*m[s].mean()), up=float(np.mean(np.abs(O[s,0])<0.3))))
  return r
out={n:per_min(n) for n in range(19200,19239)}
json.dump(out,open(sys.argv[1],'w'))
for n,r in out.items(): print(n,' '.join('%4.1f/%.2f'%(x['rng'],x['up']) for x in r[:8]))
