# 微观成网恢复与投料逆设计里程碑

固定版本 **`aam-integrated-process-20261121`**。用三张含噪局部快照推断微观成网轨迹，再从同一轨迹分布为延迟支撑和早缓冲后松弛选择投料时刻与左右比例。512 个独立模拟母批中，原任务的 18 项盲测对照全部通过事先设定的非劣门槛。恢复和设计的最不利同时单侧 95% 相对平均损失上界为 +0.696% 和 +3.290%，各自容差为 1% 和 5%。结论覆盖既定模拟器和控制域。水凝胶实测验证仍待完成。

- [技术方法与结果](evidence/AAM_INTEGRATED_PROCESS_NI_RESULTS_ZH.md)
- [预登记协议](evidence/AAM_SAME_ATLAS_INTEGRATED_PROTOCOL_ZH.md)
- [独立数值审计](evidence/INDEPENDENT_AUDIT.json)
- [原封存包和冻结源码的 SHA-256](evidence/ORIGINAL_ARCHIVE_BINDINGS.json)

## 本机复核和运行

在本目录用 Python 3.13 安装 [冻结运行依赖](requirements.txt) 后执行以下命令。

```powershell
python -B check_milestone.py --verify
python -B check_milestone.py --query data/EXAMPLE_QUERY.json
```

`source/` 是逐文件校对过的 94 份冻结源码。`data/` 保留运行恢复和规划所需的模拟轨迹图集、缓存损失、评分数组、冻结工艺和联合概率的审计样例。`--verify` 从逐母批数组重算正式统计界。`--query <JSON 路径>` 可用三行四列模拟局部读数运行连续投料方案的选择。JSON 的每行依次是左侧 AAm 读数、左侧 MBAA 读数、右侧 AAm 读数、右侧 MBAA 读数，行顺序为原三个观测时刻。

原字节复现包另在本机 H 盘里程碑目录保存，包含所有原始模拟续接、登记及评审。GitHub 目录具备预测和统计复核所需数据。原始路径级独立审计应使用 H 盘完整复现包。

## 从冻结版本继续开发

在仓库中基于固定标签创建新的工作副本。新的方法改动留在开发分支，已封存目录和标签保留既有证据。

```powershell
git fetch origin --tags
git worktree add -b dev/aam-new-experiment ../time_hydrogel-aam-experiment aam-integrated-process-20261121
```
