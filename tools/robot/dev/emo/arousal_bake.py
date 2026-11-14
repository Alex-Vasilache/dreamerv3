import glob,sys,json,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score as auc
HZ=65; B='/bucket/DoyaU/vasilache/bucket/results/dreamerv3/'
def ema(x,tau_s):
    a=1/(tau_s*HZ); y=np.empty_like(x); s=x[0]
    for i,v in enumerate(x): s=(1-a)*s+a*v; y[i]=s
    return y
def zs(x,tau=10):
    m=ema(x,tau); v=ema((x-m)**2,tau); return (x-m)/np.sqrt(v+1e-12)
def bkl(b,p):
    b=np.clip(b,1e-5,1-1e-5); p=np.clip(p,1e-5,1-1e-5); return b*np.log(b/p)+(1-b)*np.log((1-b)/(1-p))
cols=['step','theta','threat','td','change','pleasure','act0','intended0','push','emp16_state','emp16_value','emp16_threat','arousal_total','arousal_kl','arousal_unexpected']
res={}
for r in sys.argv[1:]:
    f=glob.glob(B+r+'_*/logdir/emotion.csv')[0]
    d=pd.read_csv(f,usecols=cols).drop_duplicates('step',keep='last').set_index('step')
    d=d.reindex(np.arange(d.index.min(),d.index.max()+1)).ffill()
    th=np.degrees(d.theta.values); n=len(th)
    hit=np.flatnonzero(((th[1:]>6.5)&(th[:-1]<=6.5))|((th[1:]<-5)&(th[:-1]>=-5)))+1
    pu=np.flatnonzero((d.push.values[1:]!=0)&(d.push.values[:-1]==0))+1
    y=np.zeros(n,bool); yb=np.zeros(n,bool); yp=np.zeros(n,bool)
    for h in hit: y[max(0,h-16):h+16]=True; yb[max(0,h-20):max(0,h-6)]=True
    for h in pu: yp[h:h+20]=True
    m=np.zeros(n,bool); m[5*60*HZ:]=True
    T=d.threat.values.astype(float); tn=np.clip(T/(np.percentile(T[m],99.9)+1e-9),0,1)
    td=d.td.values.astype(float); a0=d.act0.values.astype(float)
    S={}
    S['yu_base_rising']=bkl(tn,ema(tn,30))*(tn>ema(tn,30))
    S['yu_base']=bkl(tn,ema(tn,30))
    for k in (8,13): S[f'imminence_{k}']=zs(np.maximum(0,T-np.roll(T,k)))
    S['stakes']=zs(d.emp16_value.values+d.emp16_threat.values)
    S['abs_td']=zs(np.abs(td))
    al=np.zeros(n); a=0.
    for i in range(n): a=0.1*abs(td[i-1] if i else 0)+0.9*a; al[i]=a
    S['pearce_hall']=zs(al)
    v=ema(td**2,5); S['precision_pe']=zs(np.abs(td)/np.sqrt(v+1e-12))
    S['effort_control']=zs(a0**2)
    S['override_control']=zs(np.abs(d.intended0.values-a0))
    for k in ['arousal_total','arousal_kl','arousal_unexpected']: S[k]=d[k].values
    S['appraisal_parts']=np.stack([zs(d.arousal_kl.values),zs(np.abs(d.change.values)),zs(np.maximum(0,np.diff(T,prepend=T[0]))),-zs(d.emp16_state.values)],1)
    X=np.stack([d.pleasure.values,np.abs(a0),np.ones(n)],1)[m]
    out={}
    for k,val in S.items():
        if k=='appraisal_parts': out[k]=val[m].tolist() if False else None; continue
        val=np.nan_to_num(val.astype(float)); mb=m&~(y&~yb)
        coef=np.linalg.lstsq(X,val[m],rcond=None)[0]; resid=val[m]-X@coef
        out[k]=[auc(y[m],val[m]),auc(yb[mb],val[mb]),auc(yp[m],val[m]) if yp[m].any() else None,auc(y[m],resid)]
    np.save(f'app_{r}.npy',np.concatenate([S['appraisal_parts'][m],y[m,None],yb[m,None],mb[m,None]],1).astype(np.float32)[::4])
    res[r]=out; print(r,{k:None if v is None else [None if x is None else round(x,2) for x in v] for k,v in out.items()},flush=True)
json.dump(res,open(f'bake_{sys.argv[1]}.json','w'))
