"""Rebuild report figures from measured output: python report/build_figures.py."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
r=json.loads((ROOT/'results/default_trajectory.json').read_text())
e=json.loads((ROOT/'results/experiments.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(8,7),layout='constrained')
for key,color,style,label in [('controlled','#176b5b','-','With feedback'),('natural','#9aa99a','--','Natural recovery')]:
    axs[0].plot(r[key]['time'],r[key]['distance'],style,color=color,lw=2,label=label)
axs[0].axvspan(3,5,color='#f6e6cd',alpha=.8,label='Perturbation')
axs[0].axhline(.05,color='#c99442',ls=':',label='Recovery threshold')
axs[0].set(xlabel='Time (illustrative days)',ylabel='Aitchison distance',title='Default perturbation and recovery experiment')
axs[0].legend(fontsize=8)
shares=np.array(r['controlled']['shares'])
for i,(name,color) in enumerate(zip(['Vata','Pitta','Kapha'],['#7298ad','#cf865b','#82936c'])):
    axs[1].plot(r['controlled']['time'],shares[:,i],color=color,label=name,lw=2)
    axs[1].axhline(r['controlled']['baseline'][i],color=color,ls=':',alpha=.55)
axs[1].set(xlabel='Time (illustrative days)',ylabel='Composition share',title='Controlled composition and fixed reference shares')
axs[1].legend(fontsize=8)
fig.savefig(ROOT/'report/recovery.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
for ax,(name,values) in zip(axs,e['cohort_benchmark'].items()):
    cm=np.array(values['confusion_matrix']);ax.imshow(cm,cmap='Greens',vmin=0,vmax=60)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>35 else '#243b36')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=['Vata','Pitta','Kapha'],yticklabels=['Vata','Pitta','Kapha'],xlabel='Predicted class',ylabel='True class',title=name)
fig.savefig(ROOT/'report/confusion.png',dpi=180)
