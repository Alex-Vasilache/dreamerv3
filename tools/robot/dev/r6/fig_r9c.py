import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; S=sys.argv[1]
R=json.load(open(S+'/r9c.json'))
C={'control':'#000000','half':'#56B4E9','full':'#D55E00'}
L={'control':'live recipe','half':'smoothing ×0.5 (quad 0.25, temporal 0.15)','full':'smoothing (quad 0.5, temporal 0.3)'}
fig,ax=plt.subplots(2,1,figsize=(7,5.2),sharex=True)
for a,runs in R.items():
  M=min(len(v) for v in runs.values()); x=np.arange(1,M+1)
  for k,j in ((0,1),(1,2)):
    Y=np.array([[v[i][j] for i in range(M)] for v in runs.values()]); m,s=Y.mean(0),Y.std(0)
    ax[k].plot(x,m,color=C[a],label=L[a]); ax[k].fill_between(x,m-s,m+s,color=C[a],alpha=0.15,lw=0)
ax[0].set_ylabel('bumper crossings / min'); ax[1].set_ylabel('wobble (tilt-rate SD, deg/s)'); ax[1].set_xlabel('minute of training (table, sim)')
ax[0].legend(frameon=False,fontsize=8); ax[0].set_title('Round 9c: action smoothing dose, n=4 per arm',fontsize=10)
for a_ in ax: a_.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig(S+'/rpt2/figs/r9c.png',dpi=130)
