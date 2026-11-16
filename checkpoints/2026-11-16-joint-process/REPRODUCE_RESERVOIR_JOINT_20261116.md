# 复现联合逆设计二乘二试验

从 [研究说明](RESERVOIR_JOINT_PROCESS_ZH.md) 和 [图](reservoir_joint_figures/JointProcess.png) 开始。联合实验使用 Python 3.13.5、NumPy 2.1.3、SciPy 1.15.3、Numba 0.61.0，图件另需 Matplotlib 3.10.0。公开数据复算另用 openpyxl 3.1.5。`MANIFEST_SHA256.json` 用于校验整个包内每个载荷。原有参考任务的协议保留在原相对路径，供新实验核对哈希。

## 模拟二乘二和冻结输出

从解压目录打开 PowerShell，用新输出目录复算。`--workers` 可按本机能力调整。冻结的样本不需要任何外部私有或公共资料。

```powershell
$prior = 'runs/patch_reservoir_joint_20261116T120340314306Z'
$new = 'runs/joint_replay'
New-Item -ItemType Directory $new
Copy-Item -LiteralPath "$prior/REGISTERED.json" -Destination "$new/REGISTERED.json"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
python -B patch_reservoir_joint_study.py development $new --workers 4
python -B patch_reservoir_joint_study.py freeze $new --workers 4
python -B patch_reservoir_joint_study.py score $new --workers 4
python -B audit_patch_reservoir_joint.py $new
python -B audit_patch_reservoir_joint_bank.py $new
python -B render_reservoir_joint.py "$new/SCORED.json" reservoir_joint_figures/Replay
```

`REGISTERED.json` 指定数据划分、观测、候选、数值约束、盲区对照与时刻选择。`DEVELOPMENT.json` 固定同活度早时点。`FROZEN.json` 保存原始测量、后验权重、选材、失效概率与过程区间。`SCORED.json` 保存两档网格的候选结果、实际合成初态和固定主样轨迹。`INDEPENDENT_SCORE_AUDIT.json` 从保存值独立回算达标、遗憾、Brier、覆盖、母批配对、库存守恒和盲区上限。`BANK_INDEPENDENT.json` 是独立的非零界面阻力 Radau 对照。

另有冻结后诊断 `interval_support_audit/`。将其登记复制到新目录，并运行 `python -B audit_reservoir_joint_support.py [新目录]`。该登记引用包内原结果，以区分候选支持不足和后验等尾区间内的失配。

## 公开时空剖面的复算

[Xu 等 Geltrex 实测资料](external/xu_geltrex_transport/XU_GELTREX_ZH.md) 给出原发布链接、文件哈希、工作簿拆分和复现步骤。作者工作簿可从其官方数据源取得。包内提供来源收据、冻结输出、评分、负结果和独立 XML／Radau 数值复算。一个实测体系中的留出时刻预测和两个合成情景的联合逆设计分别报告。

## 实物交接

沿用 [时间任务模板](RESERVOIR_PHYSICAL_TASK_TEMPLATE.json)。正式运行前需确定同母批标识、内部水化的测量量与平台、库存化验回收率、浴液和界面校准、主样的电学与力学用途边界、独立留出位置和时间。当前包的合成条目数量不能用来设定独立试验样本数。
