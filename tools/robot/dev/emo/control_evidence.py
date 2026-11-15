"""Evidence-based control estimate on DC runs: sliding-window regression of the
tilt acceleration proxy (second difference of theta) on the wheel command.
control = R^2 (share of tilt change explained by my commands) and slope."""
import sys, glob, json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
B='/bucket/DoyaU/vasilache/bucket/results/dreamerv3/'
def auc(lo_in, out):  # AUC of LOW value marking "inside"
  u=mannwhitneyu(out, lo_in).statistic; return u/(len(lo_in)*len(out))
res={}
for n,g in [(44981,.2),(44982,.5),(44983,.2),(44984,.5),(44985,.2),(44986,.5),(44987,.2),(44988,.5)]:
  f=glob.glob(B+f'e{n}_*/logdir/emotion.csv')[0]
  d=pd.read_csv(f,usecols=['t','theta','act0','pain','control_cut','dominance'])
  d=d[d.t>d.t.iloc[0]+300].reset_index(drop=True)  # from minute 5
  d=d[d.t<d.t.iloc[0]+1500].reset_index(drop=True)
  th=d.theta.values; a=d.act0.values
  y=np.r_[th[2:]-2*th[1:-1]+th[:-2],np.nan,np.nan]
  cut=(d.control_cut.values!=0)
  pain=d.pain.fillna(0).values>0
  near=np.convolve(pain.astype(float),np.ones(33),'same')>0
  out={}
  for W in (65,130,325,650):
    s=pd.Series
    k=lambda x: s(x).rolling(W,min_periods=W//2).mean().values
    ma,my=k(a),k(y); cov=k(a*y)-ma*my; va=k(a*a)-ma**2; vy=k(y*y)-my**2
    slope=-cov/np.maximum(va,1e-9); r2=cov**2/np.maximum(va*vy,1e-12)
    for nm,v in (('slope',slope),('r2',r2)):
      ok=np.isfinite(v)&~near
      # measure state 1-5 s into the cut vs outside
      into=cut&ok; o=~cut&ok
      out[f'{nm}_W{W}']=auc(v[into],v[o])
  dm=d.dominance.ffill().values; ok=~near&np.isfinite(dm)
  out['dominance']=auc(dm[cut&ok],dm[~cut&ok])
  res[n]=dict(gain=g,**out)
  print(n,g,{k:round(v,2) for k,v in out.items()},flush=True)
json.dump(res,open('ev.json','w'),indent=1)
df=pd.DataFrame(res).T
print(df.groupby('gain').agg(['mean','std']).T.round(2).to_string())
