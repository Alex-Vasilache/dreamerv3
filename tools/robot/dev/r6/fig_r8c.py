import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
E='/work/DoyaU/vasilache/work/sysid/r6/ev'
fl=[('t','fm025',-0.25),('t','fit',0),('t','f025',0.25),('t','f050',0.5),('c','f075',0.75),('c','f100',1.0),('c','f120',1.2)]
arms=[('r6_mixed (e8261)',['fl8_%s/e8261@838500'],'#000000'),('8c, floors −0.25..1.35',['r8c_%s/e%d@682500'%('%s',n) for n in (19392,19394,19395)],'#0072B2'),('8c, floors 0.3..1.35',['r8c_%s/e%d@682500'%('%s',n) for n in (19390,19391,19393)],'#E69F00')]
fig,ax=plt.subplots(2,1,figsize=(7,5),sharex=True)
for lab,pats,c in arms:
  for k,key in ((0,'corr_f'),(1,'dips_per_min')):
    Y=np.array([[json.load(open(f'{E}/{p%s}_0.json'))[v][key] for s,v,x in fl] for p in pats])
    x=[x for s,v,x in fl]; ax[k].plot(x,Y.mean(0),'o-',color=c,label=lab)
    if len(Y)>1: ax[k].fill_between(x,Y.mean(0)-Y.std(0),Y.mean(0)+Y.std(0),color=c,alpha=0.15,lw=0)
    Yc=[json.load(open(f'{E}/{(p%"c")}_0.json'))['fit'][key] for p in pats]
    ax[k].plot([1.0],[np.mean(Yc)],'s',color=c,ms=7,mfc='none')
ax[0].set_ylabel('tracking (corr)'); ax[1].set_ylabel('falls / min'); ax[1].set_xlabel('floor (0 = fitted table / your lab, 1 = carpet; square = fitted carpet)')
ax[0].legend(frameon=False,fontsize=8); ax[0].set_title('Round 8c: continuing r6_mixed on wider floors (n=3 per arm)',fontsize=10)
for a in ax: a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('r8c.png',dpi=130)
