"""Self-contained Chinese report, with interpretation written after full curves."""
import json,csv
from exploration_common import RUN,ROOT,read_json
from report_exploration import table,figure,fmt,esc


def main():
    p=read_json(RUN/'ppo_analysis.json');wm=read_json(RUN/'wm/trend_analysis.json');budget=read_json(RUN/'compute_ledger.json');audit=read_json(RUN/'artifact_validation.json');interpret=read_json(RUN/'interpretation.json');diag=read_json(RUN/'training_diagnostics.json');names={'uniform_rsi':'均匀随机相位','start50':'50% 起点 + 50% 随机相位','original':'原始 20,480 步','uniform_rsi_repeat':'均匀组最终重放','start50_repeat':'起点组最终重放'}
    title='SuperDex：固定设置下延长学习趋势'
    b=f'<header><p>2026-10-27 · clip-000000 / wooden spoon · 单环境 · 物理 GPU 0</p><h1>{title}</h1><p>承接 b93903f；PPO 从各自策略和优化器续训，世界模型从同一初始化方案重新匹配训练。冻结数据，保留失败，不以学会作为停止条件。</p></header>'
    b+='<section><h2>完整曲线后的判断</h2>'
    for q,answer in interpret['judgments']:b+='<h3>'+esc(q)+'</h3><p>'+esc(answer)+'</p>'
    b+=f'<p><strong>预算实耗 {budget["total_charged_seconds"]/60:.2f} 分钟 / {budget["gpu_hours"]:.3f} GPU·小时，上限 60 分钟。</strong>预算剩余未用于追加实验；未运行第二种子、奖励/控制修改或解析残差结构对照。</p></section>'
    b+='<section><h2>PPO：匹配续训与固定评估</h2><p>uniform_rsi 与 start50 分别恢复上轮最终策略、Adam 优化器和 CUDA 随机状态；各自已累计 61,440 个训练控制步。继续种子 17，每组追加最多 81,920 步，每 20,480 步保存并评估。仿真从一次新重置开始，因为旧检查点不含完整物理运行中状态。两组唯一比较因素仍是重置分布，网络、动作尺度、奖励、优化设置与物理参数未改动。</p><p>均匀组相位为 0–41；start50 一半概率固定在 0，另一半均匀采样。每 128 个控制步执行 4 次 PPO 更新，γ=0.99、λ=0.95、学习率 3×10⁻⁴、clip=0.2。手腕位置伺服与 20 个手指关节不变；物体仅在重置时赋状态，此后通过真实仿真动力学和接触运动。</p><p>最终统一评估用同一个物理计算图、原来的 8 份初始 qpos/qvel。既包括上一轮检查点，也包括本轮检查点，避免把来自不同计算图的差异当成学习趋势。两个最终策略各再重放一次，用于暴露数值重复差异，不作为第二种子。</p>'
    rows=[]
    for r in p['curve']:rows.append([names[r['arm']],r['total_training_steps'],fmt(r['mean_return'],3),fmt(r['mean_length']),f'{r["successes"]}/8',fmt(r['first_failure_median_s']),r['failed_trials']])
    b+=table(['策略','累计控制步','平均回报','平均执行步','严格成功','失败回合首次越界中位数/s','失败回合数'],rows)
    b+='<p>严格成功仍要求 51 个控制步均保持位置误差 ≤30 mm、朝向误差 ≤30°，并且没有触发原失败条件。完整执行不等于严格成功。首次失败时间对失败回合统计；没有越界的成功回合按 1.02 s 右删失单列，不人为赋予失败时间。</p>'
    b+=figure('ppo_complete_curve','累计训练曲线，竖虚线是本轮起点；严格成功和回报分开显示。位置/朝向按每个初始条件的公共执行区间比较。')
    b+='<h3>逐检查点公共区间</h3><p>每个初始条件取两组实际执行长度的最小值，在相同前缀内计算误差，再对 8 个初始条件等权平均。均值变小不能通过提前终止得到。所有逐回合失败类型、首个越界时间、终止原因和回合长度见 fixed_evaluation.csv。</p>'
    rows=[]
    for pair in p['matched_checkpoints']:
        c=pair['comparison']
        for r in c['means']:rows.append([pair['total_training_steps'],names[r['arm']],fmt(c['mean_common_seconds'],3),fmt(r['position_mean_mm']),fmt(r['rotation_mean_deg'])])
    b+=table(['累计控制步','策略','公共区间均长/s','位置均误差/mm','朝向均误差/°'],rows)
    b+='<h3>本轮开始与结束共同前缀</h3>'
    c=p['start_and_final_common'];b+=table(['策略','累计控制步','四策略公共区间均长/s','位置/mm','朝向/°'],[[names[r['arm']],r['total_training_steps'],fmt(c['mean_common_seconds'],3),fmt(r['position_mean_mm']),fmt(r['rotation_mean_deg'])] for r in c['means']])
    b+='<h3>最终检查点逐回合失败</h3><p>提前终止原因码：0=执行到片段末尾，1=位置超过150 mm，2=朝向超过1.4 rad，3=关节越界，4=非有限值。原因码0并不代表严格成功。</p>'
    final_steps={r['arm']:r['total_training_steps'] for r in p['final_common']['means']};records=list(csv.DictReader((RUN/'fixed_evaluation.csv').open()))
    b+=table(['策略','初始条件','首次严格失败/s','首个越界类型','执行步数','提前终止原因码'],[[names[r['arm']],r['trial'],r['first_strict_failure_s'],r['first_strict_failure_type'],r['steps'],r['termination_reason']] for r in records if r['arm'] in final_steps and int(r['total_training_steps'])==final_steps[r['arm']]])
    b+='<h3>训练诊断</h3>'
    b+=table(['策略','新增步数','Adam 前→后步数','末40次 KL','PPO clip比例','动作截断比例','优势标准差','价值解释方差'],[[names[r['arm']],r['steps'],str(r['optimizer_steps_before'])+' → '+str(r['optimizer_steps_after']),fmt(r['diagnostics']['approx_kl']['last40_mean'],4),fmt(r['diagnostics']['ppo_clip_fraction']['last40_mean'],3),fmt(r['diagnostics']['action_clip_fraction']['last40_mean'],3),fmt(r['diagnostics']['advantage_std']['last40_mean'],3),fmt(r['diagnostics']['explained_variance_before_update']['last40_mean'],3)] for r in diag])
    b+=figure('ppo_diagnostics','原始更新值与后向 20 次更新均值。训练分布不同，训练奖励不能直接作为固定初始条件性能。')
    b+='<p>'+esc(interpret['ppo_diagnostics'])+'</p></section>'
    b+='<section><h2>世界模型：固定数据下增加优化量</h2><p>本轮不混入新增 PPO 日志，也不额外采集缺失类别。dataset.npz、完整回合划分、验证采样索引、SDF 网格和运动学元数据均从上轮逐字节复制并核对 SHA-256。保留 W0/W1-mini 的 3×256 SiLU 结构、种子 23、64 个窗口批次、8 步自回归损失、物体权重 4 和学习率 2×10⁻⁴。双方共享公共权重初始化，W1 的额外 84 维 SDF/梯度输入列初始为零；每次更新使用同一窗口批次序列。</p>'
    b+=f'<p>实际完成 <strong>每个模型 {wm["updates"]} 次更新</strong>。保留 0、250、500、590、750、1000、1250、1500、1750、2000 等已到达节点的 8/32 步验证结果；590 节点专门与上一轮等训练量比较。模型从真实初始状态递归预测，后续不输入真实状态，但使用日志中实际执行的未来控制，因此是条件预测，不是模型策略闭环。</p>'
    b+=table(['模型','同为590次更新时与上轮 state_dict 最大差'],[[r['model'],fmt(r['max_state_dict_difference_at_same_590_updates'],10)] for r in wm['previous_590_reproduction']])
    b+=figure('wm_learning_curves','同时观察训练目标、留出回合验证目标、自然分布与手部参与子集的 8/32 步误差。虚线为上一轮停止的 590 次更新。')
    b+='<h3>590 次更新与本轮末尾</h3>'
    selected=[r for r in wm['metrics'] if r['model'] in ['W0','W1'] and r['update'] in [590,wm['updates']] and r['subset'] in ['natural','hand']]
    b+=table(['模型','更新','预测步数','子集','窗口','回合','位置P50/mm','位置P90/mm','位置均值/mm','朝向P50/°'],[[r['model'],r['update'],r['horizon'],r['subset'],r['windows'],r['episodes'],fmt(r['position_p50_mm']),fmt(r['position_p90_mm']),fmt(r['position_mean_mm']),fmt(r['rotation_p50_deg'])] for r in selected])
    b+='<h3>接触分组与简单基线</h3><p>接触分组沿用最后子步的实际 GPU 负载接触（单点法向力 >0.01 N），按无接触、手部、桌面、两者模式判断。持续接触是模式保持，切换是模式改变，不代表逐指接触恒定。32 步无接触验证窗口为零，明确 N/A；8 步无接触只有 10 个窗口。自然分布和手部参与子集按类别总体/抽样数加权；分组内按窗口等权。</p>'
    b+=figure('wm_contact_regime_curves','各接触阶段的完整趋势与保持状态、常速度、重力基线。y 轴为对数，范围不同；缺失类别不填零。')
    selected=[r for r in wm['metrics'] if (r['model'] not in ['W0','W1'] or r['update']==wm['updates']) and r['subset'] in ['no_contact','sustained_contact','contact_switch']]
    b+=table(['模型','预测步数','阶段','窗口','位置P50/mm','位置P90/mm','位置均值/mm','朝向P50/°'],[[r['model'],r['horizon'],r['subset'],r['windows'],fmt(r['position_p50_mm']),fmt(r['position_p90_mm']),fmt(r['position_mean_mm']),fmt(r['rotation_p50_deg'])] for r in selected])
    b+=table(['集合','回合','转移','无接触/%','手部参与/%','仅桌面/%'],[[r['split'],r['episodes'],r['transitions'],fmt(100*r['no_any_contact_fraction']),fmt(100*r['hand_contact_fraction']),fmt(100*r['floor_only_fraction'])] for r in wm['data_statistics']])
    ct=budget['world_model_component_training_seconds']
    b+='<p>'+esc(interpret['wm_limits'])+f'</p><p>W0 192,064 参数，W1 213,568 参数。训练组件耗时分别为 {ct["W0"]:.1f} s 和 {ct["W1"]:.1f} s，匹配更新数和数据不等于匹配计算量。只有一个片段、一个物体、一个训练种子；验证窗口在同一留出回合内可能重叠，不可把窗口数当作独立重复。没有新增 yaw 规范化、物理上下文变化或模型梯度策略耦合；不是正式 G1。</p></section>'
    b+='<section><h2>算力账目、有效性与边界</h2>'
    b+=table(['部分','上限/min','实际 GPU 进程/min'],[[k,fmt(v/60),fmt(budget['charged_seconds'][k]/60)] for k,v in budget['allocation_seconds'].items()])
    b+=table(['进程','实耗/s','硬上限/s','退出码','是否硬超时'],[[r['name'],fmt(r['wall_seconds']),r['hard_cap_seconds'],r['returncode'],r['hard_timeout']] for r in budget['processes']])
    b+='<p>账目按监督进程实际观测的子进程墙钟时间，包含 Python 导入、GPU 计算、该进程内 CPU/I/O 和周期评估；并非 CUDA 内核时间、FLOPs 或能耗。物理始终串行单世界。超时保护只针对本次启动的进程组，不操作其他任务。CPU 离线报告、哈希核验与绘图不计作 GPU 进程时间。</p>'
    b+=table(['实际检查','结果'],[[k,'通过' if v else '未通过'] for k,v in audit['checks'].items()])
    b+='<p>状态和重置的原有实现未改；离线核对固定初始状态、执行控制、奖励、成功判据、首次失败、优化器步数与冻结数据哈希。没有重跑旧的静态保持或完整参考成功门槛，也没有开放式工程诊断。批量敏感性仍未解决，限制单环境；现有结论不支持直接扩大任务或宣称训练已达理论上限。</p></section>'
    b+='<section><h2>复现与交付</h2><pre>ssh root@47.100.227.205 -p 9091\ncd /lost+found/nvme/dexmimic/superdex\nsource activate_mjw.sh\nexport SUPERDEX_EXPLORE_RUN="$PWD/runs/learning_trend_20261027"\n# 只核对已有产物，CPU 执行\npython scripts/validate_learning_trend.py\n# 全流程在新的目录复现，有明确时间/步数上限\nbash scripts/reproduce_learning_trend.sh</pre><p>产物包括两组 PPO 的新状态—动作日志、中间/最终检查点、世界模型中间/最终检查点、冻结数据与验证窗口、8/32 步预测、训练 CSV、固定评估、PNG/PDF 图表、算力账目、源代码及 Git bundle。新 PPO 数据单独保存，供后续数据版本使用，本轮没有并入世界模型数据。</p><p>物理重跑依赖工作站现有 .venv-mjw 及父安装的 Wuji 资产；独立环境与原 Isaac Gym 环境均保留。压缩包不重复包含原始 HOT3D 大 tar 或父安装，但包含处理后的参考和碰撞资产、冻结模型数据、上轮源检查点及本轮全部学习产物。</p><p class="small">图表使用 scientific-visualization 技能，保留原始值、明确单位/缺失值/平滑方式并进行视觉核验。工具方法参考：Kassis et al. (2026), <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a>。该文不是本轮结果证据。</p></section>'
    css='body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#183040;background:#f1f5f7;margin:0}main{max-width:1220px;margin:auto;padding:28px 20px}header,section{background:white;padding:28px;margin:0 0 22px;border-radius:12px}header{border-top:8px solid #0072b2}h1{font-size:30px}h2{font-size:23px;color:#005d91}h3{margin-top:25px}table{border-collapse:collapse;font-size:13px;margin:15px 0;min-width:85%}th,td{padding:9px 11px;border-bottom:1px solid #ccd8df;text-align:left;white-space:nowrap}th{background:#eaf2f7}.scroll{overflow-x:auto}img{width:100%;height:auto}figure{margin:24px 0}figcaption,.small{font-size:13px;color:#526372}pre{padding:20px;background:#eef2f5;overflow-x:auto}strong{color:#9c3e00}a{color:#006ca8}'
    (RUN/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+css+'</style><main>'+b+'</main></html>',encoding='utf-8');print('Saved report.html')


if __name__=='__main__':main()
