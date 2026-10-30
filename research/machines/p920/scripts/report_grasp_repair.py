"""Compile diagnostic evidence without converting episode completion to success."""
import json,csv,hashlib,html,subprocess,collections
from pathlib import Path
import numpy as np
from grasp_lab import RUN,OLD,ROOT,DATA,write_json,VARIANTS

def load(p):return json.loads(p.read_text())
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def agg(rows):
    return dict(n=len(rows),successes=sum(r['dynamic_success'] for r in rows),horizon_completions=sum(r['reason']=='horizon' for r in rows),mean_length=float(np.mean([r['steps'] for r in rows])),mean_return=float(np.mean([r['return_'] for r in rows])) if 'return_' in rows[0] else None,reasons=dict(collections.Counter(r['reason'] for r in rows)),trials=rows)


def evaluation():
    base=RUN/'gpu_original_graph';labels=['baseline_zero','baseline_checkpoint','grip_balance_checkpoint'];runs={x:agg(load(base/x/'summary.json')) for x in labels};pairs={}
    for key,a,b in [('checkpoint_control_ablation','baseline_checkpoint','grip_balance_checkpoint'),('current_zero_vs_checkpoint','baseline_zero','baseline_checkpoint')]:
        trials=[]
        for i in range(8):
            x=load(base/a/f'trial_{i:02d}_metrics.json');y=load(base/b/f'trial_{i:02d}_metrics.json');n=min(len(x),len(y))
            trial=dict(trial=i,common_controls=n,common_seconds=n*.02,baseline_position=float(np.mean([r['position_m'] for r in x[:n]])),candidate_position=float(np.mean([r['position_m'] for r in y[:n]])),baseline_rotation=float(np.rad2deg(np.mean([r['rotation_rad'] for r in x[:n]]))),candidate_rotation=float(np.rad2deg(np.mean([r['rotation_rad'] for r in y[:n]]))))
            for label,folder in [('baseline',a),('candidate',b)]:
                z=np.load(base/folder/f'trial_{i:02d}.npz');trial[label+'_common_return']=float(z['reward'][:n].sum());trial[label+'_common_reward_components']=z['reward_components'][:n].sum(0).tolist()
            trials.append(trial)
        pairs[key]=dict(trials=trials,means={k:float(np.mean([r[k] for r in trials])) for k in trials[0] if k not in ['trial','baseline_common_reward_components','candidate_common_reward_components']},meaning='Same execution prefix per trial, equally weighted across eight trials. Full horizon success is separate.')
    reproduction=[]
    for i in range(8):
        suffix='' if i==0 else f'_{i:02d}';a=np.load(OLD/f'rollout_after{suffix}.npz');b=np.load(base/'baseline_checkpoint'/f'trial_{i:02d}.npz')
        assert np.array_equal(a['qpos'][0],b['qpos'][0]) and np.array_equal(a['qvel'][0],b['qvel'][0])
        candidate=np.load(base/'grip_balance_checkpoint'/f'trial_{i:02d}.npz')
        reproduction.append(dict(trial=i,identical_initial_state=True,first_action_max_delta=float(abs(a['raw_action'][0]-b['action'][0]).max()),candidate_first_action_max_delta=float(abs(candidate['action'][0]-b['action'][0]).max()),old_controls=len(a['raw_action']),new_controls=len(b['action']),old_return=float(a['reward'].sum()),new_return=float(b['reward'].sum())))
    result=dict(engine='GPU MuJoCo Warp, original 10-substep graph, GPU actor',success_rule='51 controls from phase zero, EVERY sample <=0.03 m and <=pi/6 rad; original failure rules unchanged.',runs=runs,pairs=pairs,baseline_reproduction=reproduction,new_training=False)
    write_json(RUN/'fixed_evaluation.json',result)
    with (RUN/'fixed_evaluation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['label','trial','steps','return_','strict_success','first_strict_failure_s','termination_s','reason']);w.writeheader()
        for label in labels:
            for r in runs[label]['trials']:w.writerow(dict(label=label,trial=r['trial'],steps=r['steps'],return_=r['return_'],strict_success=r['dynamic_success'],first_strict_failure_s=r['first_strict_violation_s'],termination_s=r['termination_time_s'],reason=r['reason']))
    return result


def preserve():
    original=ROOT/'runs/real_track_20261026/delivery/artifact_manifest.json';rows=[]
    if original.exists():
        for entry in load(original)['files']:
            p=entry['path'];dest=ROOT/'runs/real_track_20261026'/p[len('results/'):] if p.startswith('results/') else DATA/p[len('processed/'):] if p.startswith('processed/') else None
            if dest is not None:
                digest=hashlib.sha256(dest.read_bytes()).hexdigest();rows.append(dict(path=str(dest.relative_to(ROOT)),unchanged=digest==entry['sha256'],sha256=digest))
        assert rows and all(r['unchanged'] for r in rows),'Original baseline artifact changed.'
    write_json(RUN/'baseline_preservation.json',dict(original_commit='0667e7d39dbdc07b47c90a92f7031ba04f284443',verified_against_original_delivery_manifest=bool(rows),files=rows,all_unchanged=bool(rows) and all(r['unchanged'] for r in rows)))
    return len(rows)


def main():
    ev=evaluation();preserved=preserve();paired=load(RUN/'paired_common_prefix.json');events=load(RUN/'events_and_failures.json');batch=load(RUN/'batch_evidence_summary.json');learning=load(RUN/'learning_design_audit.json');geometry=load(RUN/'collision_geometry_audit.json');wrench=load(RUN/'wrench_audit.json')
    ablations=[]
    for f in sorted(RUN.glob('cpu_paired_*.json')):
        rows=load(f)
        for variant,task in sorted(set((r['variant'],r['task']) for r in rows)):
            z=[r for r in rows if r['variant']==variant and r['task']==task];a=agg(z)
            ablations.append(dict(variant=variant,task=task,engine='cpu',successes=sum(r['static_hold_2s_within_10mm_15deg'] if task=='A' else r['dynamic_success'] for r in z),length=a['mean_length'],horizon=a['horizon_completions'],parameter=VARIANTS[variant]))
    write_json(RUN/'ablation_summary.json',ablations)
    # Every root-cause statement includes its evidence and what it does not prove.
    causes=[
        ['确认并修复（本轮评估器）','候选策略观察量混合精度','交付前发现候选重建观察量混用float64参考与float32状态，首动作差异约1.49e-8至5.96e-8；改为原始float32减法，并仅重跑受影响的8个候选。修正后初始动作与基线逐位一致。','observation_dtype_correction.json；gpu_original_graph/grip_balance_checkpoint_mixed_dtype/；fixed_evaluation.json','这是本轮新增评估器的问题，不是已证明的旧PPO训练失败根因。旧候选记录保留但不计入最终表。'],
        ['确认并修复','接触日志错位 2 ms','实际接触属于最后积分前的状态；新采样保存该状态、动作和 GPU 接触力，CPU 在同一时刻重建。','batch_*/cpu_comparison.json；*_contact_aligned_metrics.json','只修复诊断归属，不会自动改善抓握。'],
        ['本轮未发现错误','世界索引、状态拷贝、重置范围与流','2/8 世界首尾重置的全部状态与历史字段其他行不变；零拷贝地址一致，共用 CUDA 流。','reset_markers_2.json / reset_markers_8.json','一步重放有浮点差异；不声称所有实现细节已被穷尽验证。'],
        ['确认现象，根因未完全定位','GPU 接触/求解数值敏感','8 世界首次非逐位差异 2 ms；76 ms 接触集合分叉，78 ms 物体差异超过 1 µm；最大 25.91 mm。单世界微扰也放大，CPU 对照未同等放大。','batch_8/first_difference_evidence.json；gpu_microperturb.json；cpu_microperturb.json','不能直接称为真实物理混沌，也不能据此排除 GPU 碰撞或求解实现问题。'],
        ['强支持的失败机制','初始接触布局与预载不能稳定托住物体','A 静态零速度、手腕准确保持，仍 0/8；扰动后的初始接触手部几何数量 1–4。线性接触受力可行性检查未通过。','wrench_audit.json；gpu/baseline/A/；tracking_and_contact.png','LP 只分析瞬时点接触与近似摩擦锥/力矩限值，不证明真实 Wuji 硬件或其他抓姿不可行。'],
        ['已做单因素排查，未修复','初始速度、手腕滞后、预载分布、缺少食指接触','零速度、20/35 ms 手腕前馈、condim=4、预载重分配、食指目标补全均未得到严格成功。','ablation_summary.json；paired_common_prefix.json；incremental_index_ablation.json','候选都是假设或本片段拟合参数，不能作为已测量硬件参数或已验证稳定控制。'],
        ['风险证据，因果未证实','物体参数与碰撞近似','质量40 g、摩擦1、惯量与桌面为假设。接触点至视觉网格差异中位约1.632 mm、最大约1.745 mm；原网格非水密。','collision_geometry_audit.json；scene.xml','视觉网格也不是真实接触面的计量真值；本轮未同时更换质量、摩擦、惯量和碰撞模型。'],
        ['设计风险，尚未训练消融','随机相位绕过困难起点','旧策略 CPU 名义起点于0.2 s终止；42个等权RSI相位中41个能到各自片段末尾，但这是段落完成率。','initialization_audit.json','剩余时域不同，不能把97.6%当作完整轨迹成功率；未证明改为起点训练就能解决抓握。'],
        ['确认并修复','旧训练准入条件过弱','以前有限/几何检查不足以保证抓握；新配置将训练入口置为未通过，测试验证不会创建训练输出目录。','configs/grasp_repair_gate.json；training_gate_test.json','保留0667e7d和全部基线数据；本轮无新PPO更新。']
    ]
    write_json(RUN/'root_cause_evidence.json',dict(columns=['status','issue','evidence','files','limit'],rows=causes))
    with (RUN/'root_cause_evidence.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['status','issue','evidence','files','limit']);w.writerows(causes)
    means=ev['pairs']['checkpoint_control_ablation']['means'];sections=[]
    sections.append('<h1>SuperDex：单轨迹抓握诊断与修复验证</h1><p class="lead">结论：诊断链和训练准入已修正；稳定抓握尚未修复。固定 8 个初始状态的严格成功仍为 0/8。本轮没有扩大任务数量，也没有延长或重启 PPO。</p>')
    sections.append('<p>对象：HOT3D clip-000000 / spoon_wooden / 右手 Wuji；1.02 秒、51 个控制步。基线提交 <code>0667e7d</code>。本报告由保存的 NPZ/JSON 生成；Git 提交见交付清单 <code>artifact_manifest.json</code>。所有新结果位于 <code>runs/grasp_repair_20261026</code>。</p>')
    sections.append('<h2>1. 资源与证据边界</h2><p>开始时物理 GPU 0 被另一任务占用约23.3 GiB，仅余约420–572 MiB。CPU 诊断先行；随后实测只含物理的诊断约占340 MiB，在保留至少64 MiB余量后串行完成1/2/8世界测试。低显存 A/B/C 接触诊断的 C 使用 CPU 策略推理。后来 GPU 0 其他任务自行退出、空闲约23 GiB，补做了原始10子步图与GPU策略推理的固定评估。没有停止或修改其他进程；GPU 1 未用于本项目。共享负载下的耗时和分配失败不构成算法或吞吐结论。</p>')
    sections.append(f'<p>原交付清单中的 {preserved} 个基线结果/处理数据文件已逐一核对 SHA-256，全部保持不变。8 个初始 qpos/qvel 从原 before/after 日志直接提取并验证一致；没有重采样或替换。物体只在各次重置时设置状态，执行中只写伺服控制。</p>')
    sections.append('<h2>2. 根因证据表</h2>'+table(['状态','问题','证据','对应文件（均相对新结果目录）','结论限制'],causes))
    brow=[]
    for r in batch:
        n=r['nworld'];ft=r['first_interworld_position_differences'].get('1e-06') if n>1 else None
        brow.append([n,r.get('first_nonbitwise_qpos_time_s','—'),ft['time_s'] if ft else '—',f"{r['maximum_interworld_object_position_spread_m']*1000:.3f}",f"{(1-r['different_contact_count_fraction'])*100:.2f}%",f"{r['max_total_gpu_contacts']}/{r['allocated_contact_capacity']}"])
    sections.append('<h2>3. 批量隔离与接触证据</h2>'+table(['世界数','首次非逐位 qpos 差异(s)','首次位置差异>1µm(s)','最大世界间物体差异(mm)','同状态CPU/GPU手物接触数量一致率','GPU接触峰值/容量'],brow)+'<p>第一次细小数值差异、第一次接触集合分叉和达到工程尺度的差异分开记录。<code>batch_*/first_difference_evidence.json</code> 包含对应世界的积分前后状态、速度、相同控制、实际接触点与力。1世界没有世界间差异这一指标。接触数量一致不等于几何或受力完全相同；匹配使用同几何对最近点，CPU与GPU热启动及精度不同，因此不要求力逐位一致。已安装版本的 CONTACTS 排序函数为 pass，源码副本保存在 <code>installed_determinism_source.txt</code>；本轮未擅自修改第三方求解器。</p><img src="visuals/batch_and_sensitivity.png" alt="相同世界的物体差异增长；0.1微米扰动在GPU单世界中比CPU产生更大偏差。">')
    sections.append('<h2>4. 独立静态保持与单因素消融</h2><p>A：原起点、全部速度置零、恒定控制2秒，独立诊断标准为全程10 mm/15°以内。B：原参考初速度、完整手腕和手指控制、物体自由运动。C：已有PPO检查点，无学习。动态严格成功条件保持原样：51步全部位置误差≤30 mm、转角≤30°且无失败。终止条件仍为位置>150 mm、角度>1.4 rad、关节越界或非有限值。静态诊断和动态成功率从不混用。</p>')
    names={'baseline':'原控制','contact4':'仅启用condim=4扭转摩擦','wrist_ff':'仅20ms手腕前馈','wrist_ff35':'仅35ms手腕前馈','zero_velocity':'仅动态初速度置零','grip_balance':'仅手指预载分布拟合','index_contact':'仅补食指接触目标','grip_balance_index':'在预载候选上增加食指目标'}
    sections.append(table(['CPU控制候选','任务','通过数/8','平均控制步','到达时域末尾/8'],[[names[r['variant']],r['task'],f"{r['successes']}/8",f"{r['length']:.2f}",r['horizon']] for r in ablations]))
    sections.append('<p>预载拟合只改变拇指/中指/无名指的原预载倍数，分别为3.367/0.109/0.487；食指和小指为1。用无扰动名义起点前0.4秒、243次CPU评估拟合，再检查独立2秒静态与原8个扰动状态；这仍是本开发片段调参。食指补全采用几何目标约1mm预重叠与原参考接触门控（41/52状态），不是接触标注或新的奖励。组合候选的增量比较单独保存在 <code>incremental_index_ablation.json</code>。这些候选均未被采纳为稳定抓握修复。</p><img src="visuals/paired_grasp_diagnostics.png" alt="8个静态保持回合均失败；下图为每对动态回合共同时间区间上的误差。"><img src="visuals/tracking_and_contact.png" alt="试验0的手腕和关节误差、GPU法向接触力与切向滑移代理。">')
    sections.append('<p>每个试验保存物体位姿/速度、手腕和手指误差、接触数量/位置/局部及世界力。首次滑移采用可复查代理：有载法向力>0.1 N时，力加权切向相对速度连续两次采样>20 mm/s。采样间隔20ms，接触时间为状态返回时间减2ms；它不是触觉真值或连续时间精确起点。另列相对抓握位姿偏离、第一次无几何接触、第一次严格条件违反和真正终止时间：见 <code>events_and_failures.json</code>。无接触时切向速度记0，不能解释为粘着。</p>')
    sections.append('<h2>5. 原图 + GPU策略的固定评估</h2>'+table(['执行方式','严格成功','到达51步','平均回合步数','平均回报'],[[label,f"{r['successes']}/8",f"{r['horizon_completions']}/8",f"{r['mean_length']:.2f}",f"{r['mean_return']:.4f}"] for label,r in ev['runs'].items()]))
    sections.append('<p>baseline_zero 为原参考控制（零残差），baseline_checkpoint 为原控制与已有检查点，grip_balance_checkpoint 只替换控制预载，策略权重和原策略观察保持不变。重放原检查点仍为6个角度终止、2个时域末尾，平均20.25步；返回值并非逐位重现。全部初始状态完全一致，首个动作差异和旧/新回报在 <code>fixed_evaluation.json</code> 中逐项记录。低显存分段图的C结果单独保留，不能与该表混算。</p>')
    sections.append(f'<p>原检查点与预载候选的配对共同区间：位置误差 <b>{means["baseline_position"]*1000:.2f} → {means["candidate_position"]*1000:.2f} mm</b>；角度误差 <b>{means["baseline_rotation"]:.2f} → {means["candidate_rotation"]:.2f}°</b>。每对只平均两者都实际执行的时间，之后8对等权平均。候选更长的回合可累积更高回报，不能据此认定成功；该候选仍0/8。</p>')
    sections.append(table(['试验','原检查点：步数/首次严格失败(s)/终止','预载候选：步数/首次严格失败(s)/终止'],[[i,f"{a['steps']} / {a['first_strict_violation_s']:.2f} / {a['reason']}",f"{b['steps']} / {b['first_strict_violation_s']:.2f} / {b['reason']}"] for i,(a,b) in enumerate(zip(ev['runs']['baseline_checkpoint']['trials'],ev['runs']['grip_balance_checkpoint']['trials']))]))
    sections.append('<h2>6. 对照视频</h2><p>全部来自已保存的GPU物理状态，不是强制放置物体的参考播放。固定选择试验0、同一相机；50Hz状态以25fps慢放。早停画面标明最终状态冻结。</p>'+''.join(f'<h3>{t}：{label}</h3><video controls preload="metadata" poster="visuals/comparison_{t}_000.png" src="visuals/comparison_{t}.mp4"></video>' for t,label in [('A','独立2秒静态保持'),('B','自由物体参考控制'),('C','已有检查点，原控制与预载候选')]))
    sections.append('<h2>7. 学习设计检查与未执行项目</h2><p>42个随机参考起始相位被逐一检查，与片段起点使用相同无噪声规则。41/42能到各自片段末尾，起点在0.2秒角度失败；这提示RSI可能绕开困难抓握起点，但不构成训练因果消融。没有增加接触奖励：原参考接触尚未证明稳定，且任务包含放置/释放，持续抓握奖励可能与目标冲突。</p>')
    sections.append(f'<p>GAE公式的跨回合边界与截断bootstrap例检查通过；旧训练日志未保存每次更新的值预测、旧log-prob和优势张量，无法完整回溯历史优势、clip fraction和explained variance。原训练平均/最大近似KL为{learning["mean_approx_kl"]:.4f}/{learning["max_approx_kl"]:.4f}，裁剪前梯度范数为{learning["mean_preclip_gradient_norm"]:.2f}/{learning["max_preclip_gradient_norm"]:.2f}；梯度上限0.5。保存检查点的探索标准差约5mm手腕平移、0.05rad手指动作，相对毫米敏感的抓握较大。奖励六分项已从本轮每步状态动作重建，并与原环境奖励在1e-4内逐步核对；完整和共同前缀回报均保存。有限片段结束是否应按终止还是时间截断bootstrap，仍需在下一次训练前冻结约定。</p>')
    sections.append('<p><b>本轮新增PPO更新数：0。</b>这是遵循“物理与控制修复通过后再训练”的条件；GPU资源后来释放也没有改变物理未通过的事实。训练入口已读取新准入配置，防止仅凭旧有限值/几何检查重新开跑。未运行新初始化训练消融、新动作尺度训练、接触奖励训练、正式D0或G1。</p>')
    sections.append('<h2>8. 复现与下一阶段准入</h2><pre>ssh root@47.100.227.205 -p 9091\ncd /lost+found/nvme/dexmimic/superdex\nsource activate_mjw.sh\n# 新目录；不会覆盖此次证据。CPU部分先执行。\nbash scripts/reproduce_grasp_repair.sh cpu\n# GPU 0资源足够后，串行执行；参数为上一步结果目录。\nbash scripts/reproduce_grasp_repair.sh gpu /absolute/path/to/new/run\n# 汇总与视频（EGL固定GPU 0）。\nbash scripts/reproduce_grasp_repair.sh report /absolute/path/to/new/run</pre>')
    sections.append('<p>已通过：1/2/8世界输入一致、重置隔离/地址/流检查、接触容量检查、实际GPU接触记录、时间对齐CPU重建、固定8状态提取、奖励重建、基线哈希保护、训练门禁拒绝测试。未通过：稳定2秒保持、严格动态抓握成功、批量轨迹数值一致性准入。可继续使用1/2/8世界做受控诊断；<b>不具备扩展并行训练/正式采集、进入正式D0或G1的条件</b>。下一步应在同一片段上构造并验证有受力依据的初始抓握与接触感知重定向，同时界定GPU碰撞/求解精度，之后才评估单个训练设计改动。没有来自多个任务或独立训练种子的研究结论。</p>')
    sections.append('<p class="note">主要源：<a href="https://mujoco.readthedocs.io/en/latest/mjwarp/">MuJoCo Warp官方文档</a>（GPU float与CPU double、接触实现差异；运行版本源码快照随包保存）。图形方法记录：Kassis et al. (2026), <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills</a>。使用科学可视化技能，原数据、变换、所有试验和缺失/早停语义均保留；无平滑、无置信区间、无期刊合规声明。</p>')
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SuperDex 单轨迹抓握诊断</title><style>body{max-width:1220px;margin:40px auto;padding:0 24px;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#182331;background:#fafbfd}h1{font-size:32px}h2{margin-top:42px;border-bottom:2px solid #cbd5e1}p{max-width:1100px}.lead{background:#fff1da;border-left:5px solid #a95500;padding:18px;font-size:18px}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5;background:white}td,th{padding:10px;border:1px solid #d8dee5;text-align:left;vertical-align:top}th{background:#e8edf3}img,video{max-width:100%;display:block;margin:24px auto}pre{background:#e8edf3;padding:20px;overflow:auto}code{overflow-wrap:anywhere}.note{font-size:13px;color:#42546a}a{color:#005a96}</style><body>'+''.join(sections)+'</body></html>'
    (RUN/'report.html').write_text(doc,encoding='utf-8');print('Saved report, fixed evaluation, ablations and root-cause evidence.')


if __name__=='__main__':main()
