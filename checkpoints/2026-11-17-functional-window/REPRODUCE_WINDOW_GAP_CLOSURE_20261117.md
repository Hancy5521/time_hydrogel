# 时窗研究补缺版的复现入口

[研究报告](WINDOW_GAP_CLOSURE_20261117_ZH.md) · [随机流更正](WINDOW_RANDOM_STREAM_CORRECTION_ZH.md) · [实物交接程序](WINDOW_CONFIRMATION_EXECUTION_ZH.md)

进入此检查点目录，必要时先补齐本地完整档案中的附带输入。模拟依赖 Python 3.13.5、NumPy 2.1.3、SciPy 1.15.3、Numba 0.61.0 及 scikit-learn 1.6.1。校验程序仅需 Python，开发测试另需 pytest。公开工作簿复算所需版本见各来源的单独说明。建议为 Numba 缓存保留单独目录。

## 方法与固定资产

| 接口 | 作用 | 入口 |
|---|---|---|
| 低维物理后验 | 同一早期见证片、低／高活度释放、半层库存，保留多个未来空间轨迹 | `window_spatial_probe.py`, `patch_exchange_forward.py` |
| 决策分数 | 从整窗失效概率和余量分位计算原概率、余量排序、学习得分。只从独立训练批估计系数 | `window_event_policy.py` |
| 精确风险 | 独立母批条件风险、活动率的联合检验。另支持不等价的边际事件排序校准 | `window_event_calibration.py` |
| 候选容量 | 配对的 9 对 25 目录内插，25 对 27 几何新方向及同面库存 | `patch_nested_catalogue_benchmark.py`, `patch_thickness_catalogue_benchmark.py` |
| 实物接口 | 输出两种只包含相应见证读数的方法输入，冻结时窗和方案，逐时点判定后按母批合并 | `validate_window_confirmation.py` |

所有存档中的 `REGISTERED.json` 记录用于冻结条件、母批划分的具体源码字节和上游散列。原有不可变文件既可作为视图结果也可作为新复现的依赖。不同协议中的 `test-000` 等只是本次组内索引，统一身份需要协议散列、组角色及索引。`MANIFEST_SHA256.json` 逐项标识包内文件。

## 新的独立母批风险确认

可复现目录 `runs/patch_window_strict_independent_20261117T023031718082Z`。只使用前期无复用的开发角色锁定算法和旧阈值。新母批重新采样，初次的混合队列保留诊断身份。正式复算时，新建空目录并复制登记文件。之后不覆盖原目录。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$new = 'runs/strict_replay'
New-Item -ItemType Directory $new
Copy-Item 'runs/patch_window_strict_independent_20261117T023031718082Z/REGISTERED.json' "$new/REGISTERED.json"
python -B check_window_random_streams.py "$new/REGISTERED.json" "$new/STREAM_CHECK.json"
python -B confirm_window_event_gate.py forecast $new --workers 4
python -B confirm_window_event_gate.py calibrate $new --workers 4
python -B confirm_window_event_gate.py score $new --workers 4
python -B confirm_loose_window_gate.py $new
```

原 25% 和严格 10% 条件风险规则属于两个已冻结操作点，分别使用单侧检验大小 .025。它们各自另有 10% 母批活动率要求。混合风险、分机制风险与相同早期信息的浴液机制压力结果分别记录。只以合适的角色判断置信解释。初次风险结果目录的精确置信推断过期，保留其逐母批点结果和错误记录供撤回核查。

## 同风险目标下的厚度验证

主目录 `runs/patch_window_thickness_independent_20261117T023031799865Z`。在新目录复制该登记文件，先运行同样的 `check_window_random_streams.py`。随后对 `calibrate_thickness_catalogue.py` 依次运行 `forecast`、`lock`、`calibrate`、`score`。固定面积的导电读法、模拟采样时刻、服务机制和单位面积初始离子量约束见协议。两个门槛使用同样信息及母批，候选数为 25 和 27。底层合并计算的 43 个候选共用 256 套物理参数轨迹，计算耗时随候选数和时间格变化。

原始厚度容量结果和时间加密审计在 `runs/patch_thickness_catalogue_20261117T021658657133Z`。它们用于诊断原范围不可行与新几何改变如何影响所有时间余量。`sampled_time_audit` 在原母批的更密时点上核对可行和原选择。`bath_geometry_audit` 固定原选择，检查相同浴液绝对体积及不同剂量设置。`audit_equal_size_thickness_capacity.py` 读取同一目录的冻结结果，在原母批上穷举去掉两项后报告可行上限的范围。三项均使用原母批。统计结论限于该试验采用的时点和材料模拟范围。

640 母批的重复在 `runs/patch_window_thickness_delivery_replicate_20261117T025235612415Z`。复制登记至新目录后运行 `check_window_random_streams.py`，再对 `replicate_thickness_delivery.py` 分别执行 `forecast`、`score`。它严格沿用原两套门槛和轨迹预测。另一个事前登记的学习评分位于 `runs/patch_window_geometry_score_20261117T025740037371Z`，由 `recalibrate_catalogue_score.py fit` 和 `evaluate` 复现。其训练只用前一试验的 340 个合成校准母批，不改变原先的固定门槛实验。前 320 个新母批用于新的评分校准，后 320 个用于两种规则配对比较。原门槛复验仍严格使用全部 640 个且不依赖替代规则。

## 新公开实测与真实接入

本地完整档案保留来源条款和可运行审计。出版方原件按研究报告的 DOI 获取。部分公开来源不具备共同试样身份，不强制拟合或将其当作原目标材料。

实物工作从 `WINDOW_CONFIRMATION_EXECUTION_ZH.md` 开始。合成软件演示不含新材料标签。固定真实用途阈值、采集时刻和设备后，独立母批分别记录湿与弱湿试样及导电、力学时窗。多日采集可在同一协议下逐块冻结和评分。统计样本数按独立制备母批计算。
