"""Chinese, self-contained evidence report generated from frozen result files."""
import base64,html,json,csv,collections
from exploration_common import RUN,ROOT,read_json


def esc(x):return html.escape(str(x))
def fmt(x,n=2):return 'N/A' if x is None else f'{x:.{n}f}'
def table(head,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def figure(name,caption):
    data=base64.b64encode((RUN/'visuals'/f'{name}.png').read_bytes()).decode()
    return f'<figure><img alt="{esc(caption)}" src="data:image/png;base64,{data}"><figcaption>{esc(caption)}</figcaption></figure>'


def main():
    p=read_json(RUN/'ppo_comparison.json');ledger=read_json(RUN/'compute_ledger.json');w=read_json(RUN/'wm/evaluation.json');a=read_json(RUN/'wm/prediction_analysis.json');m=read_json(RUN/'wm/dataset_manifest.json');cfg=read_json(RUN/'wm/config.json');audit=read_json(RUN/'artifact_validation.json');diagnostics=read_json(RUN/'training_diagnostic_summary.json');failures=read_json(RUN/'failure_types.json')
    full=p['comparisons']['final_arms_full_common']['means'];prefix=p['comparisons']['original_and_both_final'];components=ledger['details']['world_models']['component_training_seconds']
    names={'original':'原 20,480 步检查点','uniform_rsi':'均匀随机相位续训','start50':'50% 起点 + 50% 随机相位','original_repeat':'原检查点末尾重放'}
    title='SuperDex：预算受控的探索性学习实验'
    body=f'<header><p>2026-10-26 · HOT3D clip-000000 · wooden spoon / object 4 · 单个物理世界</p><h1>{title}</h1><p>从原检查点继续学习；保留失败证据和成功标准；本轮没有增加任务或扩大并行物理环境。</p></header>'
    body+='<section><h2>本轮结论</h2><p><strong>已实际完成两组各 40,960 步 PPO 续训和 W0/W1-mini 各 590 次配对更新。</strong>起点覆盖提高了固定评估回报、延后首次越界，并降低位置误差，但朝向误差变大，严格成功仍是 0/8。W1 在 32 步接触切换预测上有局部收益；8 步和简单动力学基线比较不支持“全面优于”。</p>'
    body+=f'<p>保守记账共 <strong>{ledger["total_charged_seconds"]/60:.2f} 分钟 / {ledger["gpu_hours"]:.3f} GPU·小时</strong>，上限 60 分钟。算力预算结束条件与“必须学会”无关；两组 PPO 都达到步数上限，世界模型因时间上限停止。未使用余额继续试验。</p>'
    body+='<p>2 秒静态保持和原参考控制成功不再作为 PPO 前置门槛。状态与重置正确性、奖励、有限数值和资源约束继续保留。没有附着物体、运行中重写物体位姿、改变成功阈值或采用上一阶段的抓握候选参数。</p></section>'
    body+='<section><h2>实验一：唯一改动是起点采样覆盖</h2><p>两组从同一原检查点恢复策略和 Adam 状态，续训种子均为 17。均匀组沿用相位 0–41 的均匀重置；start50 组一半概率强制相位 0，另一半仍均匀采样，因此实际相位 0 概率略高于 50%。动力学、控制尺度、奖励、动作探索分布、学习率和优化步数均相同。每 10,240 步保存一次检查点；总累计训练量均为 61,440 步。</p><p>每次只有一个物理世界执行。训练和周期评估使用独立单世界缓冲区，切换时同步 Torch/Warp 流。最终曲线由同一个物理图依次评估所有检查点，固定为原来的 8 份初始 qpos/qvel。原检查点在首尾各重放一次，量化单世界数值重放差异。</p>'
    rows=[]
    for r in p['learning_curve']:
        rows.append([names[r['arm']],r['total_steps'],fmt(r['mean_return'],3),fmt(r['mean_length']),f'{r["successes"]}/8',fmt(r['first_failure_median_s']),f'{fmt(r["first_failure_min_s"])}–{fmt(r["first_failure_max_s"])}'])
    body+=table(['策略','累计训练步数','平均回报','平均执行步数','严格成功','首次严格失败中位数/s','最早–最晚/s'],rows)
    body+=figure('ppo_fixed_learning_curve','同一组 8 个初始状态的固定评估。连接线只是检查点间的趋势；没有跨种子置信区间。')
    body+='<h3>共同执行时间区间，避免早停均值偏差</h3><p>两个最终策略均执行完 51 步 / 1.02 秒，以下误差覆盖完整公共区间。原检查点多次早停，与它比较时对每个初始条件截取三种策略共同执行的前缀，再对 8 个条件等权平均。</p>'
    rows=[[names[k],'两组最终策略：1.02 s',fmt(v['mean_position_mm']),fmt(v['mean_rotation_deg'])] for k,v in full.items()]
    rows += [[names[k],f'三策略公共前缀：均值 {prefix["mean_common_seconds"]:.3f} s',fmt(v['mean_position_mm']),fmt(v['mean_rotation_deg'])] for k,v in prefix['means'].items()]
    body+=table(['策略','统计区间','平均位置误差/mm','平均朝向误差/°'],rows)
    body+='<p><strong>朝向退化不能被更高回报掩盖。</strong>该片段包含向下放置过程，物体落到桌面后也可能保持较小位置误差。回合执行到末尾不等于抓握成功，策略执行也不等于参考轨迹播放。</p>'
    body+='<h3>逐回合首次失败</h3><p>严格失败：任一步位置误差超过 30 mm 或朝向误差超过 30°。环境提前终止仍沿用位置 >150 mm、朝向 >1.4 rad、关节越界或非有限状态；严格失败后可以继续执行直到终止或片段末尾。下表 0 表示未触发环境提前终止，并不表示严格成功。</p>'
    rows=[[names[r['arm']],r['trial'],fmt(r['first_strict_failure_s']),r['first_strict_failure_type'],r['termination_reason'],r['completed_controls']] for r in failures if r['arm']=='original' or (r['arm'] in full and r['additional_steps']==40960)]
    body+=table(['策略','初始条件','首次严格失败/s','首个越界类型','提前终止原因码','执行步数'],rows)
    body+='<h3>训练诊断</h3><p>GAE 使用 γ=0.99、λ=0.95；失败时不 bootstrap，正常片段结束按原实现 bootstrap，GAE 在重置边界截断。动作残差尺度为腕平移 0.025 m、腕转动 0.18 rad、手指 0.25 rad，raw action 在执行前截断到 [-1,1]。PPO 每 128 步进行 4 次更新；未增加接触奖励。</p>'
    rows=[]
    for r in diagnostics:
        d=r['diagnostics'];rows.append([names[r['arm']],fmt(100*d['reset_distribution']['phase_zero_fraction']),fmt(d['approx_kl']['last40_mean'],5),fmt(d['ppo_clip_fraction']['last40_mean'],4),fmt(d['action_clip_fraction']['last40_mean'],4),fmt(d['advantage_std']['last40_mean'],4),fmt(d['explained_variance_before_update']['last40_mean'],3)])
    body+=table(['策略','实际起点重置/%','末40次 KL均值','PPO clip比例','执行动作截断比例','GAE 标准差','价值解释方差'],rows)
    body+='<p>六个奖励分项、梯度范数和原始更新值保存在 training.csv；在线逐步检查分项之和与环境奖励一致。离线从状态与动作重新计算的最大误差为 '+fmt(audit['details']['max_reward_difference'],10)+'。训练分布不同，训练回报曲线本身不作为固定任务性能比较。</p>'
    body+=figure('ppo_training_diagnostics','浅色为原始更新数据，深色为后向 10 次更新均值；没有重复训练种子的误差带。')+'</section>'
    body+='<section><h2>实验二：W0/W1-mini 预测</h2><p>收集的是 MuJoCo Warp 真正执行的状态—控制转移，包括失败回合、短暂符合误差阈值的前缀，以及 64 个延长记录回合。延长记录在任务失败后继续物理演化，末尾保持参考手部目标，专门保留接触丢失和落桌状态；这些回合不按成功回合报告。当前没有完整严格成功回合可加入数据。</p><p>状态由 33 维 qpos、32 维 qvel 和过去 3 个执行控制组成。模型输入对四元数使用 6D 表示，输出位置/关节/速度残差及归一化旋转增量。W0 使用状态与当前控制；W1 另加 21 个手部查询点的 84 维 SDF 标量/梯度特征，几何对应仿真中的五个 CoACD 碰撞块。双方隐藏层均为 3×256 SiLU，W0 192,064 参数，W1 213,568 参数；相同公共权重初始化，W1 额外输入列初始为零，同一次更新使用相同的窗口批次。</p><p>以完整回合/初始条件族先划分，再提取窗口；2147 个训练回合与 536 个验证回合。旧日志同一初始条件族统一归属一个集合，新回合按固定哈希划分。固定 PPO 评估轨迹不参与世界模型拟合。归一化仅来自训练集；训练对可用的三种接触类别等概率抽样，物体位姿与速度损失权重为 4。验证窗口在同一回合内可能重叠，因此窗口数不能当作独立样本数。</p>'
    body+=table(['集合','回合','转移','无接触/%','手部接触/%','仅桌面接触/%'],[[s['split'],s['episodes'],s['transitions'],fmt(s['no_any_contact_fraction']*100),fmt(s['hand_contact_fraction']*100),fmt(s['floor_only_fraction']*100)] for s in m['statistics']])
    body+='<p>这里的接触是最后一个物理子步（t+18 ms）上法向力 >0.01 N 的实际 GPU 接触，下一状态在 t+20 ms。类别按“无接触 / 手部 / 桌面 / 两者”模式判断：持续接触表示模式不变，接触切换表示模式改变，不能分辨每个手指间的接触交换，也没有解析子步之间全部瞬时碰撞。</p>'
    body+=table(['集合','近似自由落体占全部/%','近似自由落体占无接触/%'],[[d['split'],fmt(d['freefall_like_fraction_all']*100),fmt(d['freefall_like_fraction_of_no_contact']*100)] for d in a['data_distribution']])
    body+='<p>近似自由落体定义为无采样接触且 50 Hz 差分线加速度距重力小于 1 m/s²。数据并非被自由落体或完全无接触主导，而是明显受桌面接触主导。连续 8 步无接触验证窗口只有 10 个，训练仅 25 个且会被过采样；32 步这一类别完全缺失，不能推断泛化能力。</p>'
    body+=figure('data_composition_and_hand_subset','接触数据分布及手部参与窗口单列。手部子集按原始验证窗口类别比例加权，避免等量分层抽样改变自然混合比例。')
    body+='<h3>8 步与 32 步预测误差</h3><p>8 步为 160 ms，32 步为 640 ms。模型从一个真实初始状态开始递归预测，后续不注入真实状态；当前实验提供日志中的未来真实执行控制，因此是条件预测，不是模型驱动策略的闭环想象。表中为预测终点位置与朝向误差，不是整个预测区间平均值。Persistence 保持状态，Constant velocity 线性/旋转积分，Ballistic 增加重力及与仿真步长一致的半隐式位置修正。</p>'
    rows=[]
    for h in ['8','32']:
        for reg in ['no_contact','sustained_contact','contact_switch']:
            for name in ['W0','W1','persistence','constant_velocity','ballistic']:
                r=w[name][h][reg];rows.append([h,reg,name,r['windows'],r['episodes'],fmt(r['position_median_mm']),fmt(r['position_p90_mm']),fmt(r['rotation_median_deg']),fmt(r['rotation_p90_deg'])])
    body+=table(['步数','阶段','模型','抽样窗口','涉及回合','位置P50/mm','位置P90/mm','朝向P50/°','朝向P90/°'],rows)
    body+=figure('world_model_regime_errors','P50 柱与 P90 标记；P90 不是置信区间。各面板范围不同且均从零起，缺失组明确显示 N/A。')
    body+='<p>32 步接触切换位置 P50：W0 56.5 mm、W1 28.0 mm；持续接触：W0 74.0 mm、W1 30.1 mm，但保持不动基线为 6.8 mm。8 步无接触：W0 39.3 mm、W1 58.3 mm，重力基线为 7.7 mm。W1 在 8 步手部参与窗口的位置 P50 与 W0 几乎相同（11.5 mm）；32 步则为 28.1 对 57.0 mm。总体不能用单一汇总值宣称胜出。</p>'
    body+=figure('world_model_learning_curve','各 590 次配对更新；训练曲线显示原始值与后向 20 次更新均值。右侧是三个类别均值的宏平均，不是自然分布加权误差，无接触小样本对它影响很大。')
    body+=f'<p>W1 训练组件耗时 {components["W1"]:.1f} s，W0 {components["W0"]:.1f} s，约 {components["W1"]/components["W0"]:.2f} 倍，主要包括当前 eager FK/SDF 计算开销。双方是匹配数据和优化更新数，不能宣称匹配模型计算量。所有采样预测均有限；CPU 重载两个最终模型并递归 8 步也已核对。</p>'
    body+='<p><strong>这不是正式 W0/W1 或 G1 复现。</strong>网络宽度缩小，只有一个物体及固定物理上下文，没有变化的 ξ/BPS、没有 yaw 规范化、没有模型梯度与策略耦合；仿真 warmstart 内部量也不在预测状态内。SDF 网格步长不超过 3 mm；CPU FK 对 MuJoCo 最大坐标差约 1.7×10⁻⁸ m，同一标量场的解析梯度、自动微分及有限差分核对通过。这只证明本次特征实现检查通过。</p></section>'
    body+='<section><h2>预算、资源和可追溯修复</h2><p>预算依据旧实验 20,480 步 / 531.1 s 的实测速率制定：60 分钟 GPU 进程时间总上限，PPO 40 分钟、补充记录 5 分钟、世界模型 10 分钟、额外评估与验证 5 分钟。该速率有共享设备负载影响，不是设备性能标定。GPU 按物理 GPU 0 UUID 固定；未修改 GPU 1 或其他进程。</p>'
    body+=table(['部分','上限/min','保守实耗/min'],[[k,fmt(v/60),fmt(ledger['charged_seconds'][k]/60)] for k,v in ledger['allocation_seconds'].items()])
    body+=f'<p>CPU 独立 SDF 构建 {ledger["cpu_sdf_build_seconds"]:.1f} s，另列而不伪计为 GPU 内核运算。总 GPU 进程时间包含该进程内 CPU、I/O 和验证等待，不等于 CUDA kernel 时间、FLOPs、能耗或独占设备吞吐。PPO 本次约 93–95 控制步/进程秒。</p>'
    body+=table(['修复','影响','处理和记账'],[['评估摘要对 CUDA 标量直接使用 NumPy','首次启动报错，尚无 PPO 更新','显式转 Python float；保留错误目录，保守记入 30 s'],['WM 归一化循环重复解压 NPZ','0 次更新，耗时异常','缓存数组后正常训练；仅终止已确认属于本实验的 PID，323 s 全额计入 WM 预算'],['SDF 查询 FK 缺少固定腕部基座变换','训练前几何实现检查失败','补齐固定变换；CPU 核对通过后才训练，旧错误数值保留'],['绘图第一条曲线提前锁定 y 轴','视觉检查发现第二条曲线被裁切','在绘制全部策略后统一设范围；未改变数据或训练结果']])
    body+='<p>保留此前接触丢失、抓握失败与批量敏感性证据。单世界首尾重放原检查点的平均回报为 1.639 与 1.667，成功仍为 0/8；这是数值重复波动，不是独立随机种子统计。批量敏感性未解决，目前继续限制为单物理世界，不阻断这种有界学习实验。</p></section>'
    body+='<section><h2>下一项假设与阶段判断</h2><p><strong>PPO：</strong>“更多起点覆盖是否稳定改善前期控制”值得一次小额第二种子复核，保持当前两组设置及固定评估；同时报告朝向退化。现有结果还不支持增加任务数或通过加长训练来保证学会。若随后研究探索幅度，应在固定起点采样下单独改变，不能把两种因素叠加后归因。</p><p><strong>世界模型：</strong>下一项更值得验证的假设是“固定解析速度/重力积分骨架 + 学习接触残差，是否减少无接触及桌面静止阶段漂移”。先冻结本轮数据划分、评估窗口与预算，和当前模型配对比较；W1 32 步局部收益值得保留，但不足以进入模型梯度策略优化。</p><p>本轮通过的是可执行训练、日志与指标核对、完整回合数据隔离、有限预测、几何特征实现和预算约束。严格抓握成功、跨种子稳健性、跨物体泛化、批量接触一致性仍未解决。可以继续单环境、有界的假设验证；<strong>不据此宣称正式 D0 / G1 通过，也不扩展并行数据采集。</strong>这不是新增“抓握准入清单”。</p></section>'
    body+='<section><h2>复现与交付索引</h2><p>工作站项目：<code>/lost+found/nvme/dexmimic/superdex</code>。父安装的 Wuji 资产与独立 .venv-mjw 是物理重跑前提；复现脚本只创建新的 runs 子目录，不覆盖原基线。学习阶段唯一新增依赖为 rtree==1.4.1，安装在 .venv-mjw 内。环境包清单和资源快照随包保存。</p><pre>ssh root@47.100.227.205 -p 9091\ncd /lost+found/nvme/dexmimic/superdex\nsource activate_mjw.sh\n# 只核对本次已保存结果（CPU，不重新训练）\npython scripts/validate_exploration.py\n# 在新目录重跑两个学习实验；最多 60 分钟 GPU 进程时间\nbash scripts/reproduce_exploration.sh</pre><p>本次真实产物位于 runs/explore_20261026：canonical_eval、两个 PPO 目录、collection、wm、visuals、compute_ledger.json、artifact_validation.json。WM 冻结数据集、回合清单、检查点、逐窗口预测、训练 CSV、固定评估 CSV、源代码快照与 Git bundle 均在交付压缩包中。原始 HOT3D 大 tar 和父 Wuji 安装不重复打包；物理复现使用现有安装。模型拟合复现可直接使用包内 dataset.npz、SDF 与运动学元数据。</p><p>本报告从结果文件生成；全部图片内嵌，原始 CSV/JSON 和独立 PNG/PDF 也保留。研究数据来源、许可、版本与哈希延续 configs/real_data_manifest.json，未重新下载或扩增任务。</p><p class="small">图表使用 scientific-visualization 技能的原始值保留、单位与样本量标注、缺失值及视觉核验流程。方法工具参考：Kassis, T., Agarwal, V., He, Y., Patel, D., &amp; Brueckner, A. M. (2026). <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a>。该文不是本实验结果的证据来源。</p></section>'
    css='body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#172b3a;background:#f1f5f7;margin:0}main{max-width:1180px;margin:auto;padding:32px 20px}header,section{background:white;margin:0 0 22px;padding:28px;border-radius:12px}header{border-top:8px solid #0072b2}h1{font-size:30px;line-height:1.3}h2{font-size:23px;color:#005d91}h3{margin-top:28px}p{max-width:1040px}table{border-collapse:collapse;font-size:13px;min-width:85%;margin:14px 0}th,td{padding:9px 12px;border-bottom:1px solid #cbd5dc;text-align:left;white-space:nowrap}th{background:#eaf2f7}.scroll{overflow-x:auto}img{width:100%;height:auto}figure{margin:24px 0}figcaption,.small{font-size:13px;color:#526372}pre{background:#eef2f5;overflow-x:auto;padding:20px}a{color:#006ca8}strong{color:#9c3e00}@media print{body{background:white}section{break-inside:auto}header,section{padding:12px}figure{break-inside:avoid}}'
    (RUN/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf-8');print('Saved self-contained report.html')


if __name__=='__main__':main()
