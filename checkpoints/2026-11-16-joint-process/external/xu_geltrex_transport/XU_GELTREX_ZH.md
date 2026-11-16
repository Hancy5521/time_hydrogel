# Geltrex 水凝胶深度剖面的留出时间预测

Xu 等在 HT-ChemoChip 的薄底部液相通道中注入 70 kDa FITC-dextran。底部浓度随入口距离和时间变化，直径约 200 µm、高度 250 µm 的 Geltrex 圆柱连接底部 SOURCE 与顶部 DRAIN。作者用荧光定标和 CLSM 获得 7、12、24 小时的深度剖面。出版方数值工作簿的 17 个横向位置为距入口 0.7–7.1 mm，间隔 0.4 mm。25 个竖向位置为距圆柱顶端 10–250 µm，间隔 10 µm。三个页签都提供 MEAN 和 SE，论文图注报告 n=3。工作簿未给出物理试样、芯片或制备母批标识。分析单位是一个几何布置下的组均值空间剖面，多个深度和几何位置不能当作独立材料。 [正式论文](https://doi.org/10.1038/s41467-018-07186-x) · [图 2](https://www.nature.com/articles/s41467-018-07186-x/figures/2) · [出版方工作簿](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-07186-x/MediaObjects/41467_2018_7186_MOESM4_ESM.xlsx)

## 观测和模型冻结

入库先读书目信息、荧光成像方法、补充方法、页签和坐标。2026-11-16 12:08:38 UTC（20:08:38 上海时间）登记后，预测代码只读取 7、12 小时的实测均值。24 小时均值在评分和独立审计才进入比较。SE、作者模拟曲线及作者拟合的浓度依赖指数均未用于拟合或评分。误差以原工作簿浓度刻度报告。图注及全文的 CC BY 4.0 授权已核对，工作簿没有发现单独许可声明。原件由入库回执记录实际 URL 和 SHA-256。

补充方法给出的横向点源边界形状为

$$g(x,t)=\frac{1000}{\sqrt{4\pi D_0t}}\exp[-x^2/(4D_0t)],\qquad D_0=57\ \mathrm{\mu m^2/s}.$$

这里 $t$ 换算为秒，$x$ 换算为 µm，1000 是由后续幅度系数吸收的计算缩放。使用论文的几何和参考横向扩散系数。Fick 对照设初始浓度为零，圆柱内竖向常系数扩散，顶部是近零汇，底部由 $g$ 驱动。$\partial_t C=D_z\partial_{zz}C$ 以 5 µm 差分和 0.05 h 线性输入推进。先验对数网格选择同一个 $D_z$，用 7、12 h 的均值拟合一个非负幅度和非负全局测量偏置。比较原位持续、以脉冲变化仅缩放 12 h 深度形状、以及采用相同脉冲但假设瞬时线性竖向剖面三种基线。前两项无需从历史估计全局系数，准稳态项使用与 Fick 相同的早期拟合数据与幅度、偏置规则。0.1856 µm²/s 是该简化模型在当前源和观测下的有效拟合值，无独立材料常数标定。

## 留出 24 小时的结果

| 预测 | 17 个位置平均剖面 MAE | 位置较持续法更准 | 各采样位置相对 12 h 的变化方向一致率 |
| --- | ---: | ---: | ---: |
| 原位沿用 12 h | 0.23676 | — | 0（不预测变化） |
| 按脉冲比例缩放 12 h 剖面 | 0.27343 | 9/17 | 70.6% |
| 竖向瞬时线性、脉冲馈入 | 0.26696 | 10/17 | 58.1% |
| **常系数竖向 Fick 扩散、脉冲馈入** | **0.20668** | **14/17** | **83.5%** |

Fick 对照相较 12 h 持续法的平均 MAE 下降 12.7%。提升主要来自 10–80 µm 与 90–160 µm，绝对误差分别由 0.09834、0.18773 下降到 0.06945、0.12711。在源侧 170–250 µm，差别很小（0.40338 到 0.39938），存在三个横向位置的结果弱于持续法。总体均值较优只说明这一物理对照在这一种材料、这一个留出时间有用。直接用时间比例缩放剖面和忽略竖向瞬态均未改善总体均值。

独立审计从 XLSX 中 OOXML 重新提取 24 h 真值、复算每个位置 MAE 与方向。用三处固定横向位置、独立 Radau 解同一有限差分系统，线性给料的最大差是 $9.62\times10^{-10}$，连续给料为 $2.26\times10^{-7}$。该数据检验一个有空间分辨的示踪物输运迟滞，不能检验双层离子库存、释离子和胶内留存的共同守恒、功能阈值或候选决策。

贴近后续材料任务的三层实验来自 Watkins、Southard、Anseth。其 PHEMA 或 PEG–PHEMA–PEG 给出了经校准的内部荧光剖面和外部累积释放，公开稿中仅见实验图点和理论曲线，Europe PMC 记录无附件、无关联数据集，Crossref 无数据关联。异质层的校正理论曲线经过原始结果拟合。该工作可为联合深度剖面与浴液流失的实验设计提供先例。 [Watkins 等原始论文](https://doi.org/10.1016/j.actbio.2006.11.006) · [作者公开稿](https://pmc.ncbi.nlm.nih.gov/articles/PMC2682343/)

## 本地复现

从本目录工作，若未附工作簿，按 `SOURCE_RECEIPT.json` 的官方 URL 下载并以 SHA-256 验证。实际登记、拟合、评分和审核均为独占创建，新的重放目录采用同一登记。

```powershell
New-Item -ItemType Directory forecast_replay
Copy-Item -LiteralPath forecast_v1/REGISTERED.json -Destination forecast_replay/REGISTERED.json
python -B compare_geltrex.py freeze forecast_replay
python -B compare_geltrex.py score forecast_replay
python -B audit_geltrex.py forecast_replay
```

收据为 `SOURCE_RECEIPT.json`、`forecast_v1/REGISTERED.json`、`FROZEN.json`、`SCORED.json`、`INDEPENDENT_AUDIT.json`。运行环境 Python 3.13.5、NumPy 2.1.3、SciPy 1.15.3、openpyxl 3.1.5。
