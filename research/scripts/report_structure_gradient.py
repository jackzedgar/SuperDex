"""Plots and self-contained report for the fixed single-seed structure/probe."""
import csv,base64,re,hashlib
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from exploration_common import RUN,read_json,write_json
from report_exploration import table,fmt,esc
C={'A':'#0072B2','B':'#B34E00','persistence':'#555555','constant_velocity':'#007856','ballistic':'#8B5AA0'}
OUT=RUN/'visuals';OUT.mkdir(exist_ok=True)
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white');fig.savefig(OUT/(name+'.pdf'),facecolor='white');plt.close(fig)
def figure(name,caption):
    return '<figure><img alt="'+esc(caption)+'" src="data:image/png;base64,'+base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()+'"><figcaption>'+esc(caption)+'</figcaption></figure>'

def main():
    s=read_json(RUN/'structure_analysis.json');g=read_json(RUN/'gradient_analysis.json');budget=read_json(RUN/'compute_ledger.json');audit=read_json(RUN/'artifact_validation.json');interpret=read_json(RUN/'interpretation.json');final=s['matched_update']
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for name,style,marker in [('A','-','o'),('B','--','s')]:
            rows=list(csv.DictReader((RUN/'wm'/name/'training.csv').open()));rr=[r for r in rows if r.get('model',name) in [name,'W1'] and r['training_loss']];x=[int(r['update']) for r in rr];y=np.array([float(r['training_loss']) for r in rr])
            axes[0,0].plot(x,y,color=C[name],alpha=.15,lw=.6);axes[0,0].plot(x[39:],np.convolve(y,np.ones(40)/40,'valid'),color=C[name],ls=style,label=name)
            vals=sorted([r for r in s['validation_loss'] if r['model']==name],key=lambda r:r['update']);axes[0,1].plot([r['update'] for r in vals],[r['normalized_8step_validation_loss'] for r in vals],color=C[name],ls=style,marker=marker,label=name)
            for ax,h,subset in [(axes[0,2],8,'natural'),(axes[1,0],32,'natural'),(axes[1,1],8,'hand'),(axes[1,2],32,'hand')]:
                rr=sorted([r for r in s['metrics'] if r['model']==name and r['horizon']==h and r['subset']==subset],key=lambda r:r['update']);ax.plot([r['update'] for r in rr],[r['position_mean_mm'] for r in rr],color=C[name],ls=style,marker=marker,label=name);ax.set(title=f'{h}-step position / {subset}',ylabel='Weighted endpoint mean (mm)')
        axes[0,0].set(title='Training: raw + trailing 40 mean',ylabel='Normalized weighted MSE',yscale='log');axes[0,1].set(title='Held-out 8-step objective',ylabel='Equal-regime normalized MSE',yscale='log')
        for ax in axes.flat:ax.set_xlabel('Update');ax.legend()
        fig.suptitle('Seed 23; A = reused W1-mini, B = two-half acceleration residual\nSame data / batches / windows / 213,568 parameters. Original reward and frozen policy.',fontsize=12);save(fig,'structure_learning')
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for i,h in enumerate([8,32]):
            for j,subset in enumerate(['no_contact','sustained_contact','contact_switch']):
                ax=axes[i,j];ax.set(title=f'{h} steps / {subset}',xlabel='Update',ylabel='Endpoint position P50 (mm; log)')
                selected=[r for r in s['metrics'] if r['horizon']==h and r['subset']==subset]
                if not any(r['windows'] for r in selected):ax.set_axis_off();ax.text(.5,.5,'N/A: zero validation windows',ha='center',transform=ax.transAxes);continue
                for name,ls,m in [('A','-','o'),('B','--','s')]:
                    rr=sorted([r for r in selected if r['model']==name],key=lambda r:r['update']);ax.plot([r['update'] for r in rr],[r['position_p50_mm'] for r in rr],color=C[name],ls=ls,marker=m,label=name)
                for name,ls in [('persistence',':'),('constant_velocity','-.'),('ballistic',(0,(4,2,1,2)))]:
                    v=next(r for r in selected if r['model']==name);ax.axhline(v['position_p50_mm'],color=C[name],ls=ls,label=name)
                ax.set_yscale('log');ax.legend(fontsize=7)
        fig.suptitle('Frozen contact strata / simple prediction baselines\nNo-contact 8-step n=10; 32-step absent. Windows overlap within held-out episodes.',fontsize=12);save(fig,'structure_contact')
        fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
        for i,name in enumerate(['A','B']):
            for j,h in enumerate([1,4]):
                ax=axes[i,j];rows=[r for r in g['details'] if r['model']==name and r['horizon']==h and r['direction']=='plus']
                for status,marker in [('improves','o'),('harms','x'),('inconclusive','D')]:
                    rr=[r for r in rows if r['status']==status]
                    if rr:ax.errorbar([r['predicted_change'] for r in rr],[r['true_change'] for r in rr],yerr=[r['noise_threshold'] for r in rr],fmt=marker,color=C[name],mfc='none' if status=='inconclusive' else C[name],ms=6,capsize=2,alpha=.85,label=f'{status}: {len(rr)}')
                ax.axhline(0,color='#555555',lw=.8);ax.axvline(0,color='#555555',lw=.8);ax.set(title=f'{name} / {h}-step suggested direction / n={len(rows)}',xlabel='Predicted reward change',ylabel='Mean real reward change');ax.legend(fontsize=8)
        for j,h in enumerate([1,4]):
            rr=[r for r in g['details'] if r['horizon']==h and r['direction']=='plus'];xx=np.array([r['predicted_change'] for r in rr]+[0]);low=min([r['true_change']-r['noise_threshold'] for r in rr]+[0]);high=max([r['true_change']+r['noise_threshold'] for r in rr]+[0]);xp=max(np.ptp(xx)*.07,1e-5);yp=max((high-low)*.07,1e-5)
            for i in range(2):axes[i,j].set_xlim(xx.min()-xp,xx.max()+xp);axes[i,j].set_ylim(low-yp,high+yp)
        fig.suptitle('All predeclared suggested directions: predicted ascent versus real replay\nWhiskers = empirical replay threshold, NOT confidence intervals. Common axes within horizon.',fontsize=12);save(fig,'gradient_directions')
    b='<header><p>2026-10-27 · 单种子 23 · 物理 GPU 0 · 单环境</p><h1>SuperDex：解析残差结构与模型梯度试验</h1><p>冻结数据和原 W1 基线，检验预测及短时动作方向。不是正式 G1/G2，也不是抓握准入门槛。</p></header><section><h2>结果与判断</h2>'
    for title,text in interpret['judgments']:b+='<h3>'+esc(title)+'</h3><p>'+esc(text)+'</p>'
    b+=f'<p>实耗 <strong>{budget["total_charged_seconds"]/60:.2f}/60 分钟 GPU 进程时间</strong>。达到更新和案例目标后停止，没有追加第二种子或完整策略训练。</p></section>'
    b+='''<section><h2>结构参数化与一致积分</h2><p>A 直接复用上一轮严格对应的种子、批次、数据、窗口、检查点及预测。B 保持 3×256 SiLU、64 维输出、213,568 参数。物体 12 个输出改作前后两个 10 ms 的线速度/角速度增量残差，其余手部预测不变。每半区间用 5 个 2 ms 子步更新同一速度与位姿，所有输出均参与计算。</p><pre>h=0.010 s, δ=0.002 s, Δ=0.020 s
每半区间的每个子步：
  v ← v + δg + r_v/5
  ω ← ω + r_ω/5
  p ← p + δv
  q ← normalize(q ⊗ Exp(δω))
等价线性公式：
  v_next = v + Δg + r_early + r_late
  p_next = p + Δv + 0.5Δ(Δ+δ)g + 0.016r_early + 0.006r_late</pre><p>位置与速度来自同一分段恒定加速度过程，不是两个独立输出加重力。线速度为世界坐标，角速度为物体局部坐标，见 <a href="https://github.com/google-deepmind/mujoco/blob/main/doc/overview.rst">MuJoCo 官方说明</a>；四元数积分与本机 mj_integratePos 核对。先验不含接触求解和完整陀螺动力学，这些效应仍由残差学习。</p><p>仅输出归一化槽 26:32、58:64 改为训练回合反解的两个半区间增量均值/标准差，下限 0.01 m/s 或 rad/s。角度反解只是小角度归一化代理；实际前向使用完整四元数组合。输入、控制、损失、其余输出归一化不变。8 步自回归、物体权重 4、学习率 0.0002、批量 64、采样和初始网络权重均保持原方案。由于物体输出参数化不同，初始物体预测不要求一致。</p></section>'''
    b+='<section><h2>固定 8/32 步预测</h2><p>最后层初始权重为零，但反归一化后的物理残差等于训练集均值，并非零残差。纯重力实现核对显式将物理残差设为零；不把未训练 B 称为纯重力模型。</p>'+figure('structure_learning','训练保留原始点及后向 40 次均值；验证不平滑。自然/手部子集按原类别总体加权。')
    selected=[r for r in s['metrics'] if r['model'] in ['A','B'] and r['update']==final and r['subset'] in ['natural','hand','no_hand']]
    b+=table(['模型','更新','步数','子集','窗口/回合','位置均值/mm','位置P50/mm','位置P90/mm','朝向P50/°'],[[r['model'],r['update'],r['horizon'],r['subset'],f'{r["windows"]}/{r["episodes"]}',fmt(r['position_mean_mm']),fmt(r['position_p50_mm']),fmt(r['position_p90_mm']),fmt(r['rotation_p50_deg'])] for r in selected])
    b+=figure('structure_contact','接触阶段与固定简单基线。32 步无接触为 N/A，没有补造数据。')
    selected=[r for r in s['metrics'] if (r['model'] not in ['A','B'] or r['update']==final) and r['subset'] in ['no_contact','sustained_contact','contact_switch']]
    b+=table(['模型','步数','阶段','窗口','位置P50/mm','位置P90/mm','位置均值/mm','朝向P50/°'],[[r['model'],r['horizon'],r['subset'],r['windows'],fmt(r['position_p50_mm']),fmt(r['position_p90_mm']),fmt(r['position_mean_mm']),fmt(r['rotation_p50_deg'])] for r in selected])
    b+=table(['集合','回合','转移','无接触/%','手部/%','仅桌面/%'],[[r['split'],r['episodes'],r['transitions'],fmt(100*r['no_any_contact_fraction']),fmt(100*r['hand_contact_fraction']),fmt(100*r['floor_only_fraction'])] for r in s['data_statistics']])
    b+='<p>完整回合划分、窗口字节不变，未混入新 PPO 数据。接触标签仅用于原采样及评估分组，不输入模型或指导动作优化。窗口在同一回合内可能重叠，不是独立重复。</p></section>'
    b+='''<section><h2>冻结模型与策略的梯度方向试验</h2><p>训练前固定初始 ID 0、2、6、7 × 前缀长度 0、5、7、10 × 1/4 步时域，共 32 例。0/2 是原训练初始条件家族，6/7 是留出家族。每次从原初始条件重置，重放相同过去控制前缀，重建求解器历史；不在测试时刻直接写入 qpos/qvel。</p><p>冻结上一轮 uniform_rsi 最终策略与 A/B。策略只在当前锚点给出原动作，随后短时保持；优化不使用真实未来状态、未来策略动作或未来接触标签。对原奖励之和计算一次梯度，按最大绝对分量归一化，扰动 L∞≤0.05，沿用原动作与控制限幅。保存原动作、建议及反方向，不调步长、不选择性保留结果。</p><p>每例原动作 5 次，A/B 正反方向各 3 次，共 17 次，固定随机顺序交错执行；建议在未来真实仿真前保存。固定执行 1/4 步，保留原终止标志；该短时奖励探测不计算抓握成功，也不靠提前终止缩短比较区间。</p><pre>ΔR_real = 方向重复均回报 − 原动作重复均回报
阈值 = max(1e-6, 基线全范围, 基线半范围 + 方向半范围)
|ΔR_real| ≤ 阈值：无法判断
超出阈值：区分改善/损害，并比较预测符号</pre>'''
    b+=table(['模型','时域','方向','案例','改善','损害','无法判断','一致/可判断'],[[r['model'],r['horizon'],r['direction'],r['cases'],r['improves'],r['harms'],r['inconclusive'],f'{r["direction_agrees"]}/{r["direction_judgable"]}'] for r in g['aggregate']])
    b+=figure('gradient_directions','全部建议方向的预测及真实奖励变化；阈值是经验重放范围，不是置信区间。反方向和全部案例见下表。')
    b+=f'<p>保留 {g["completed_cases"]}/32 例、{g["replay_count"]} 次重放。基线回报全范围最小/中位/最大值：{g["baseline_return_spread"]["min"]:.3g} / {g["baseline_return_spread"]["median"]:.3g} / {g["baseline_return_spread"]["max"]:.3g}。CPU 双精度有限差分仅验证模型求导实现，不替代真实方向效果。</p><h3>位置误差更小但建议有害</h3>'
    b+=table(['案例','模型','时域','位置/mm','另一模型/mm','朝向/°','另一模型/°','预测ΔR','真实ΔR','阈值'],[[r['case'],r['model'],r['horizon'],fmt(r['position_error_mm']),fmt(r['other_position_error_mm']),fmt(r['orientation_error_deg']),fmt(r['other_orientation_error_deg']),fmt(r['predicted_change'],6),fmt(r['true_change'],6),fmt(r['noise'],6)] for r in g['lower_position_error_but_harmful_proposal']])
    b+='<p>这里比较原动作的局部位置预测，不代表整体模型精度或朝向同步改善。</p><p>'+esc(interpret['reward_components'])+'</p><p>'+esc(interpret['replay_limits'])+'</p><h3>全部方向结果</h3>'
    b+=table(['案例','初始ID','前缀','时域','模型','方向','预测ΔR','真实ΔR','阈值','判断','控制最大变化'],[[r['case'],r['initial_id'],r['prefix_controls'],r['horizon'],r['model'],r['direction'],fmt(r['predicted_change'],6),fmt(r['true_change'],6),fmt(r['noise_threshold'],6),r['status'],fmt(r['control_delta_max'],6)] for r in g['details']])+'</section>'
    b+='<section><h2>算力、有效性与交付</h2>'+table(['类别','上限/min','实际/min'],[[k,fmt(v/60),fmt(budget['charged_seconds'][k]/60)] for k,v in budget['allocation_seconds'].items()])+table(['进程','实耗/s','退出码','硬超时'],[[r['name'],fmt(r['wall_seconds']),r['returncode'],r['hard_timeout']] for r in budget['processes']])
    b+='<p>A 的历史训练不重复记账。本轮训练及预测验证均已计入对应子进程墙钟时间，含导入和 CPU/I/O；GPU 0 与其他任务共享，不能将耗时解释为算法效率。CPU 离线分析和绘图不计 GPU 时间。仅控制本轮进程，未修改其他任务。</p>'
    b+=table(['有效性检查','结果'],[[k,'通过' if v else '未完成/未通过'] for k,v in audit['checks'].items()])
    b+='''<p>只有一个片段、物体和种子；没有第二种子、完整模型辅助策略训练或正式 G2。批量接触敏感性仍未解决。预测、短时动作方向与严格抓握成功分别报告。</p><pre>ssh root@47.100.227.205 -p 9091
cd /lost+found/nvme/dexmimic/superdex
source activate_mjw.sh
export SUPERDEX_EXPLORE_RUN="$PWD/runs/structure_gradient_20261027"
python scripts/validate_structure_gradient.py
# 新目录有界复现
bash scripts/reproduce_structure_gradient.sh</pre><p>交付包括冻结数据和 A 来源、B 中间/最终权重、曲线、全部梯度建议及物理重放、CSV、图表、算力账目和 Git bundle。物理复现依赖工作站父级 Wuji 资产及独立环境。</p><p>图表方法参考：Kassis et al. (2026), <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a>。该引用不是效果证据。</p></section>'''
    css='body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#183040;background:#f1f5f7;margin:0}main{max-width:1220px;margin:auto;padding:28px 20px}header,section{background:white;padding:28px;margin:0 0 22px;border-radius:12px}header{border-top:8px solid #0072b2}h1{font-size:30px}h2{font-size:23px;color:#005d91}table{border-collapse:collapse;font-size:13px;margin:15px 0;min-width:85%}th,td{padding:9px;border-bottom:1px solid #ccd8df;text-align:left;white-space:nowrap}th{background:#eaf2f7}.scroll{overflow-x:auto}img{width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:13px;color:#526372}pre{padding:20px;background:#eef2f5;overflow-x:auto}a{color:#006ca8}'
    (RUN/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SuperDex structure and gradient</title><style>'+css+'</style><main>'+b+'</main></html>')
    metadata=[]
    for f in sorted(OUT.glob('*.png')):
        im=Image.open(f);pdf=f.with_suffix('.pdf').read_bytes();txt=pdf.decode('latin1');assert '/Subtype /Type3' not in txt and '/FontFile2' in txt;metadata.append(dict(name=f.stem,pixels=list(im.size),dpi=list(im.info.get('dpi',[])),pdf_media_box=re.search(r'/MediaBox\s*\[([^]]+)\]',txt).group(1),png_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),pdf_sha256=hashlib.sha256(pdf).hexdigest()))
    write_json(RUN/'figure_metadata.json',dict(figures=metadata,raw_points_retained=True,training_smoothing='Trailing 40 mean, raw retained',whiskers='Empirical range threshold, not CI',missing='32-step no-contact N/A'));print('Report and three figure pairs saved')

if __name__=='__main__':main()
