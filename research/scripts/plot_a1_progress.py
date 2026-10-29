"""Unsmoothed current full-A1 collector curve; no missing result shown as zero."""
import argparse,json,csv,hashlib,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from a1_data_sources import ROOT,RUN,write
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default='collector_full_v1_001');a=p.parse_args();source=RUN/a.run
 raw=(source/'learning_curve.jsonl').read_bytes();lines=raw.splitlines();rows=[]
 for line in lines:
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:break
 assert rows
 controls=rows[-1]['real_controls'];out=RUN/'analysis'/f'collector_{controls:08d}';out.mkdir(parents=True,exist_ok=False)
 evals=[]
 for file in sorted(source.glob('evaluation_????????.json')):
  d=json.loads(file.read_text())
  if d['controls']<=controls:evals.append(d)
 (out/'source_learning_curve.jsonl').write_bytes(b'\n'.join(lines[:len(rows)])+b'\n')
 with (out/'collector_curve.csv').open('w',newline='') as f:
  keys=['iteration','real_controls','mean_rollout_reward','actor_loss','value_loss','entropy','approx_kl','clip_fraction','preclip_gradient_norm','action_saturation_fraction','elapsed_seconds'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in rows])
 with (out/'development_success.csv').open('w',newline='') as f:
  keys=['controls','regular_success_rate','strict_success_rate','n'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in evals])
 style={'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'}
 with plt.rc_context(style):
  fig,ax=plt.subplots(3,1,figsize=(180/25.4,170/25.4),layout='constrained',sharex=True)
  x=np.array([r['real_controls'] for r in rows])/1000
  ax[0].plot(x,[r['mean_rollout_reward'] for r in rows],color='#0072B2',marker='.',ms=4,lw=1,label='Rollout mean; no smoothing')
  ax[0].set(ylabel='Reward / control step',title='Full A1 shared PPO-Sim collector — ongoing');ax[0].legend(loc='best')
  for key,label,color,marker,ls in [('regular_success_rate','Regular','#0072B2','o','-'),('strict_success_rate','Strict','#D55E00','s','--')]:
   ax[1].plot([e['controls']/1000 for e in evals],[100*e[key] for e in evals],color=color,marker=marker,linestyle=ls,label=label,ms=4)
  ax[1].set(ylabel='Full-trajectory success (%)',ylim=(-2,102),title='Fixed development conditions: 40 tasks × 4 conditions');ax[1].legend(loc='upper left')
  ax[2].plot(x,[r['approx_kl'] for r in rows],color='#0072B2',marker='.',ms=4,lw=1,label='Approximate PPO KL')
  ax[2].set(xlabel='Actual training control steps (thousands)',ylabel='Approximate KL');ax[2].legend()
  for axes in ax:axes.grid(axis='y',alpha=.2);axes.set_xlim(left=0)
  fig.savefig(out/'collector_curve.png',dpi=300,transparent=False);fig.savefig(out/'collector_curve.pdf',transparent=False);plt.close(fig)
 im=Image.open(out/'collector_curve.png')
 write(out/'figure_provenance.json',dict(source_run=str(source.relative_to(ROOT)),source_curve_snapshot_sha256=hashlib.sha256(raw).hexdigest(),controls_through=controls,transformation='No smoothing or selected best checkpoints; all available complete updates and full fixed-development evaluations. No extrapolated future points; no missing result replaced by zero.',replication='One shared collector policy.160 fixed perturbation conditions nested in40source tasks. Counts describe this finite evaluation set, not population confidence intervals.',purpose='Internal research progress; no journal requirements claimed',output_dimensions_mm=[180,170],raster_pixels=list(im.size),dpi=im.info.get('dpi'),matplotlib_version=matplotlib.__version__,uncertainty='No uncertainty band for the enumerated fixed-condition fraction. Future model errors use whole-episode cluster bootstrap; not row-wise independent-window CI.',alt_text='Three panels show unsmoothed training reward, all completed regular and strict development success evaluations, and PPO approximate KL against actual physical training controls. Formal A1 model comparison is pending.',software_reference='Kassis, T.; Agarwal, V.; He, Y.; Patel, D.; Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065 . Metadata verified2026-10-28, current arXiv revision2Sep2026.'))
 print(out,flush=True)
if __name__=='__main__':main()
