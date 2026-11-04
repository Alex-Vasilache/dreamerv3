import numpy as np,glob,sys,os
# per minute: % of time in stretches where the net drive stays |.|>0.7 with one sign for >= 1 s ("runaway"),
# and mean |net wheel speed| (obs units; 1 = 3000 encoder units)
def segs(mask,minlen):
  idx=np.flatnonzero(np.diff(np.r_[0,mask.astype(int),0])); return [(a,b) for a,b in zip(idx[::2],idx[1::2]) if b-a>=minlen]
def stats(sess,mins=12):
  fs=sorted(glob.glob(sess+'/logdir/online_shared/experience/*.npz')) or sorted(glob.glob(sess+'/logdir/replay/*.npz'))
  if not fs: return None
  L=[np.load(f) for f in fs]; L=[d for d in L if 'drive' in d.files]
  W=np.concatenate([d['wheels'] for d in L]); A=np.concatenate([d['drive'] for d in L])
  # drop paused stretches (all-zero wheels and drive)
  live=~((np.abs(A).sum(1)==0)&(np.abs(W).sum(1)==0)); W,A=W[live],A[live]
  d=A.mean(1); m=np.zeros(len(d),bool)
  for sg in (d>0.7,d<-0.7):
    for a,b in segs(sg,65): m[a:b]=True
  n=min(len(d)//3900,mins)
  return [(100*m[i*3900:(i+1)*3900].mean(), np.abs(W[i*3900:(i+1)*3900].mean(1)).mean()) for i in range(n)]
if __name__=='__main__':
  for p in sys.argv[1:]:
    for sess in sorted(glob.glob(p+'/s0*'))+([p] if os.path.isdir(p+'/logdir') else []):
      r=stats(sess)
      if r and len(r)>=2: print('%-40s'%sess.split('work/')[-1][:40],' '.join('%3.0f/%.2f'%x for x in r))
