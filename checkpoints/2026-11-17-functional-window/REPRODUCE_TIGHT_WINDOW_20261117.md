# 紧时窗研究复算入口

[研究与结论](TIGHT_WINDOW_REVIEW_RESPONSE_ZH.md) · [实物验证](THICKNESS_PHYSICAL_CONFIRMATION_ZH.md) · [直接先例](TIGHT_WINDOW_PRIOR_ART_20261117_ZH.md)

证据按独立母批保存。每个 `REGISTERED.json` 含真实注册 UTC、角色及可核查源码散列。冻结文件先于相应模拟全窗结果文件。读数数量是每状态五项公用读数和两项见证读数。事件风险按任一已发请求违规的母批数计。三条登记分组均止于开发。其未来校准和确认仅为预先划分的随机地址，未生成实证数据。

| 研究 | 登记目录 | 消费角色 |
|---|---|---|
| 换第二读数与自适应选读数 | `runs/patch_tight_window_20261117T034850305493Z` | 112 个开发母批 |
| 分阶段监督修正 | `runs/patch_tight_phase_20261117T035501438587Z` | 前 112 个训练，另 112 个开发 |
| 薄见证片电学 | `runs/patch_tight_thin_coupon_20261117T035923687266Z` | 前 224 个训练，另 160 个开发 |

Python 3.13、NumPy、SciPy、scikit-learn、Numba 和 pytest 可用于复算。具体上游固定环境见引用的协议。下面生成新的独立审计副本，不覆盖原收据。命令从项目根目录运行。`audit_tight_window_supervised.py` 不拟合模型或调用正向求解器。

```powershell
$p = 'runs/patch_tight_window_20261117T034850305493Z'
$q = 'runs/patch_tight_phase_20261117T035501438587Z'
$r = 'runs/patch_tight_thin_coupon_20261117T035923687266Z'
python -B audit_tight_window_20261117.py development --run $p --output audit_witness_replay.json
python -B audit_tight_window_supervised.py $q development --output audit_phase_replay.json
python -B audit_tight_window_supervised.py $r development --output audit_thin_replay.json
python -B audit_tight_witness_identifiability.py $p pair_identifiability_replay.json
python -B audit_tight_pair_frontier.py $r pair_frontier_replay.json
python -B summarize_tight_coupon_dev.py $r thin_descriptive_replay.json
python -B -m pytest -q -p no:cacheprovider test_patch_fixture_confirmation.py test_window_confirmation.py test_window_event_calibration.py test_window_group_calibration.py
```

从头复算某条研究时，把该目录的 `REGISTERED.json` 复制到空目录，先运行对应主脚本 `forecast 空目录 development`，再运行 `score 空目录 development`。三个主脚本分别为 `tight_window_study.py`、`tight_window_phase_study.py` 和 `tight_window_coupon_study.py`。后两项引用早期被冻结的开发资料作为训练。模型估计在当时冻结过程内从训练母批生成。不同机器运行耗时、浮点末位与复算时间可有差异。实物评分器支持真实绝对时间和设备单位，其输入文档不使用模拟阈值。

公开资料的原始引用见 [研究报告](TIGHT_WINDOW_REVIEW_RESPONSE_ZH.md)。出版方附表留存在本地完整档案。
