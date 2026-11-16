"""Empowerment vs control cuts allowing a delay: the 'inside' window is
[onset+lag, onset+5s+lag]. AUC of a low value for inside vs far outside
(> 10 s from any cut), with and without bumper contact (+-0.25 s), and within
|theta| tertiles (is the drop just the body being tilted further?)."""
import glob, json, numpy as np, pandas as pd, sys
from scipy.stats import mannwhitneyu
import os
B = os.environ.get('RUNS', '.') + '/'
def auc(i,o):
  return np.nan if len(i)<50 or len(o)<50 else mannwhitneyu(o,i).statistic/(len(i)*len(o))
runs=[(n,0.5,'DC') for n in (44982,44984,44986,44988)]+[(n,0.2,'DC') for n in (44981,44983,44985,44987)]+\
     [(n,0.5,'DE') for n in (44990,44992,44994,44996)]+[(n,0.35,'DE') for n in (44991,44993,44995,44997)]
LAGS=[0,0.5,1,1.5,2,2.5,3]
out=[]
for n,g,rd in runs:
  f=glob.glob(B+f'e{n}_*/logdir/emotion.csv')[0]
  d=pd.read_csv(f,usecols=['t','theta','act0','pain','control_cut','dominance','empowerment'])
  d=d[d.t>d.t.iloc[0]+300].reset_index(drop=True)
  hz=65; cut=d.control_cut.values!=0
  dom=d.dominance.ffill().values; emp=d.empowerment.ffill().values
  near=np.convolve((d.pain.fillna(0).values>0).astype(float),np.ones(33),'same')>0
  on=np.flatnonzero(np.diff(cut.astype(int))==1)+1
  dist=np.full(len(d),1e9)
  for i in on: 
    idx=np.arange(max(0,i-650),min(len(d),i+5*hz+650)); dist[idx]=0
  far=dist>0
  T=d.theta.values; A=d.act0.values; Y=np.r_[T[2:]-2*T[1:-1]+T[:-2],np.nan,np.nan]
  k=lambda x: pd.Series(x).rolling(130,min_periods=65).mean().values
  ma,my=k(A),k(Y); slope=-(k(A*Y)-ma*my)/np.maximum(k(A*A)-ma**2,1e-9)
  th=np.abs(d.theta.values-np.nanmedian(d.theta.values))
  for L in LAGS:
    inside=np.zeros(len(d),bool)
    for i in on: inside[i+int(L*hz):min(len(d),i+int((5+L)*hz))]=True
    r=dict(run=n,gain=g,round=rd,lag=L)
    for nm,v in (('dom',dom),('emp',emp),('slope',slope)):
      ok=np.isfinite(v)
      r[nm+'_all']=auc(v[inside&ok],v[far&ok])
      r[nm+'_nocontact']=auc(v[inside&ok&~near],v[far&ok&~near])
      # within |theta| tertiles, contact excluded, then averaged
      q=np.nanquantile(th[~near],[1/3,2/3]); bins=np.digitize(th,q); a=[]
      for b in range(3):
        m=(bins==b)&ok&~near; a.append(auc(v[inside&m],v[far&m]))
      r[nm+'_theta']=np.nanmean(a)
    r['contact_in']=float(near[inside].mean())
    out.append(r)
df=pd.DataFrame(out); df.to_csv('lag.csv',index=False)
pd.set_option('display.width',200)
print(df.groupby(['gain','lag'])[['emp_nocontact','dom_nocontact','slope_nocontact','contact_in']].agg('mean').round(2).to_string())
print(df.groupby(['gain','lag'])[['emp_nocontact','slope_nocontact']].agg('std').round(2).to_string())
