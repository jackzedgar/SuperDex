"""Render the approved-scope budget proposal without changing experiments."""
import json,pathlib,html
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'configs/a1_full_budget_20261028.json').read_text())
rows=[('实现与完整路径校准',10),('Probe-40 采集策略训练',70),('约1200万条物理转移采集',100),('W0/W1各3种子匹配训练',50),('固定评估、简单基线与审计',10),('最多两轮有依据的修正',90)]
table=''.join(f'<tr><td>{name}</td><td>{h}</td></tr>' for name,h in rows)
s=p['storage'];b=p['compute_formulas'];old=p['previous_iteration']
body=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SuperDex 完整 A1：预算提案</title>
<style>body{{max-width:1000px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui,sans-serif;color:#172438;background:#fff}}h1{{font-size:30px}}h2{{font-size:22px;margin-top:32px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ccd6df;text-align:left;padding:9px}}.status{{padding:18px;background:#fff3cf;border-left:5px solid #946200}}code{{overflow-wrap:anywhere}}li{{margin:7px 0}}.small{{font-size:14px;color:#526170}}</style>
<h1>SuperDex 完整 A1：预算与执行方案</h1><p>2026-10-28 · Lambda · 提案，尚未批准新增预算</p>
<div class="status">本轮目标改为验证“接触条件下世界模型可预测”。原P/S训练保持暂停；正式A1未启动。申请新增上限<strong>330 GPU进程小时</strong>，其中首轮完整对照240小时、最多两轮修正90小时；申请<strong>60 GB可用工作空间</strong>。这些是上限与资源估计，不是获得正结果的保证。</div>
<h2>先把“完整”和“强正向”说清楚</h2>
<p>保留原Probe-40：稳定抓取搬运、手内重定向或换握、双手交接、轻物体工具使用，各10条。根据数据、参考质量与语义选取，训练前冻结名单和排除原因。当前只有两个单手物体，尚未具备这40条任务，双手环境也仍需实现。</p>
<p>用PPO-Sim约20%、50%、80%开发成功率的三个策略阶段加噪采集，目标约1200万条真实物理转移。每条轨迹、每个策略阶段约10万条；真实完整回合边界优先。实际达不到某个策略质量档时明确报告，不能把低质量策略改名为80%。</p>
<ul><li>8步物体位置/朝向误差中位数 ≤3 mm /3°。</li><li>接触切换8步误差P90 ≤8 mm /8°。</li><li>至少80%的32步窗口位置误差 ≤15 mm。</li></ul>
<p>强正向证据要求三个模型训练种子均有支持，给出各任务层、物体、接触相位和手部参与子集，以及按完整回合聚类的区间。主指标采用开放环自回归的第8/32步端点误差，另外报告沿途平均与最差误差，不能用平均掩盖末端漂移。W1优于W0与两者绝对达标是两个结论；A1通过也不代表控制梯度或A2通过。</p>
<h2>预算计算</h2><table><tr><th>阶段</th><th>GPU进程小时上限</th></tr>{table}<tr><th>合计</th><th>330</th></tr></table>
<p>原实验保留已用{old['gpu_hours_used_including_budget_calibration']:.3f}小时和剩余{old['gpu_hours_remaining']:.3f}小时，不自动叠加到新账本。此次吞吐测量实际{old['new_timing_only_gpu_seconds']:.3f}秒，已计入旧账本。</p>
<p>既有单手采集实测53.6–54.3控制步/秒。1200万条全按单手速率约{b['all_single_hand_12m_at_measured_mean_gpu_hours']:.2f}GPU小时；规划按75%单手50步/秒、25%双手25步/秒，则{b['planned_9m_single_at50_plus3m_double_at25_gpu_hours']:.2f}小时，采集阶段上限100小时。双手速率是保守假设，须在完整路径完成后替换为实测。</p>
<p>3×2048网络吞吐代理测量：512个8步窗口，W0 {b['measured_proxy_seconds_per_microbatch']['W0']:.4f}秒，W1 {b['measured_proxy_seconds_per_microbatch']['W1']:.4f}秒。每次优化累积20480窗口、每模型2000更新、共六模型，约{b['all_six_proxy_gpu_hours']:.2f}GPU小时；乘1.5的完整路径余量后约{b['with_allowance_gpu_hours']:.2f}小时，因此预留50小时。该测量仅用于算力估计：使用原单手递推、代表性输入宽度及两组SDF查询，并非完整双手A1训练。</p>
<p>四卡一直满载的纯算术下限为82.5小时；数据准备、采集策略质量、CPU重定向与任务依赖会延长墙钟时间。这不是完成日期承诺。采集策略阶段最多800万控制步或70GPU小时，以先到者为准；若仍缺高成功率采集策略，本轮范围未完整满足，停止并报告缺口，不无限延长。</p>
<h2>保证比较有解释价值</h2><ol>
<li>补齐单/双手存在掩码、物体几何与物理上下文、正确的当前手腕偏航坐标变换；W0/W1同为3×2048，唯一主变量为SDF查询。使用同种子、公共参数初始化、批次顺序、损失和更新数。</li>
<li>完整回合/初始状态家族先分70/15/15训练、开发、锁定测试，再取窗口；哈希去重。物理控制、执行器内部状态与历史均由当前或过去数据构造。记录求解器预热状态以诊断预测底噪。</li>
<li>接触相位由真实GPU接触、力和切向相对速度定义，区分自由接近、建立、持续、断开、滑移。未来接触标签不能输入预测器；缺失相位不造数据补齐。</li>
<li>每个预定检查点都保留；持久状态、匀速、重力基线用同一窗口。按自然分布和任务平衡分别汇总，保留支撑接触与手部接触的区别。</li>
<li>若首轮未达标，最多两轮单因素修正，各≤45GPU小时，例如切换窗口2倍权重、随后结构化力/力矩模型。仅使用开发集决定；原六组结果保留。配置选定前不打开最终测试结果。</li></ol>
<p>旧数据只有89621条转移，约63%仅有桌面支撑；验证集无接触8步窗口10个，32步为0。这解释了为什么不能只延长原小网络训练就宣称完成A1。旧W1/解析残差与P/S负结果均保留。</p>
<h2>空间与已执行清理</h2>
<p>仅Lambda：已删除{s['cleanup_deleted_cache_entries']}个超过60天、明确来自可重新下载wheel、无环境软链接或活动引用、无外部硬链接的缓存条目。实际释放{s['cleanup_observed_bytes_released']/1e9:.2f}GB，当前剩余{s['free_bytes_now']/1e9:.2f}GB；距离60GB目标还差{s['shortfall_to60gb_bytes']/1e9:.2f}GB。未删除任何实验结果、数据、检查点或现有环境。</p>
<table><tr><th>容量用途</th><th>GB</th></tr><tr><td>原始候选片段与来源记录</td><td>8</td></tr><tr><td>网格、碰撞、SDF</td><td>4</td></tr><tr><td>唯一转移数据、完整回合与上下文</td><td>16</td></tr><tr><td>六模型、优化器和预定检查点</td><td>6</td></tr><tr><td>两轮修正与采集策略证据</td><td>8</td></tr><tr><td>固定预测、接触证据、视频</td><td>3</td></tr><tr><td>临时空间与安全余量</td><td>15</td></tr></table>
<p>没有假设有损压缩或压缩比。历史数据只保留一份；控制历史按回合重建；不同时保留多份全量压缩包和解压图像。磁盘剩余低于5GB则安全保存并停止新增输出。</p>
<p>需要具体指定去留的旧材料：<code>/home/youran/robo-demo-hetero-scaling/results</code>约57.34GB；<code>/home/youran/old_netket_codes/tomography_data</code>约26.75GB。目录名或年龄不足以确认其中成果可删除，因此尚未处理。</p>
<h2>当前交付与批准边界</h2>
<p>预算机器可读配置：<code>configs/a1_full_budget_20261028.json</code>；测量与清理证据：<code>runs/a1_budget_20261028/</code>；计算脚本：<code>scripts/plan_a1_full_budget.py</code>。新增预算未经确认不启动正式训练。所有部署、存储、计算均限定Lambda，跳板仅转发SSH。</p>
<p>下一步需要批准新增330GPU进程小时，并落实Lambda上的60GB工作空间。收到批准后在该总额内连续完成实现、采集、六模型对照、必要修正与统一报告，不在每个小检查后重新等待提示；到预算或资源边界停止。</p>
<p class="small">依据：用户提供的《SuperDex研究计划》第5、7、12–13、20页及《算法细节》，以仓库原始PDF/文本为准。原文不是已验证实现；本方案明确记录阈值解释、实现缺口和估算假设。所有单位为十进制GB及累计GPU进程小时。</p></html>"""
(ROOT/'reports/a1_full_budget_20261028.html').write_text(body,encoding='utf-8')
readme=ROOT/'README.md';oldtext=readme.read_text()
section="""## Current plan: full A1 budget proposal (2026-10-28)

The user redirected the paused P/S iteration toward the original A1 predictability study, and asked for a concrete additional compute/storage budget before formal execution. See [the Chinese proposal](reports/a1_full_budget_20261028.html) and [machine-readable scope](configs/a1_full_budget_20261028.json). Proposed NEW cap: **330 cumulative GPU-process hours** (240 for the initial complete comparison, up to90 for at most two revisions), requiring **60GB available working space on Lambda**. This cap is **not yet approved**; no formal A1 job has started.

Probe-40 (four strata of ten), approximately12M physical transitions from three PPO-Sim quality stages, W0/W1 each three training seeds, locked episode-level splits and original G1 thresholds are required. Existing two single-hand objects and prior miniature predictions do not fulfill that scope. A positive result is a research target, not a guaranteed stopping condition. Old P/S checkpoints remain paused at commit18631dd.

Two bounded throughput-only proxy measurements consumed24.034GPU-process seconds, charged to the prior24-hour ledger; no research model was retained. Authorized old public-wheel cache cleanup freed6.876GB; all environments, experiment data and checkpoints remain. Full receipts and timing limitations are under runs/a1_budget_20261028. No other compute node was accessed.

"""
if '## Current plan: full A1 budget proposal (2026-10-28)' not in oldtext:
 lines=oldtext.splitlines(keepends=True);readme.write_text(lines[0]+'\n'+section+''.join(lines[1:]))
print('report and README written')
