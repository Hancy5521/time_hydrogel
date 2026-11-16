# 水凝胶成胶过程设计

给定局部、早期读数，模拟不同入口供料工艺下的空间成胶过程，并根据目标成胶顺序和终态选择工艺。

## 研究记录

- [双口扩散与不可逆成胶](checkpoints/2026-11-15-reactive-ports/README.md) 固定两侧各自的累计剂量，比较 17 种分流时间表。包含冻结协议、全部对照、独立数值审计和复现命令。结果来自反应–扩散模拟，材料尚未进行实测确认。

## 2026-11-16 联合过程逆设计归档

这一版同时处理释放动力学、初始空间储备和界面输运的不确定性。两组新合成情景中，双活度加半层库存相较同活度异时加总库存，联合时窗达标数由 276/512、263/512 提高至 335/512、299/512。每组含 16 个合成母批。深度区间仍有明显欠覆盖，联合材料时窗等待实物确认。

- [方法、数学、结果与实物验证边界](checkpoints/2026-11-16-joint-process/RESERVOIR_JOINT_PROCESS_ZH.md)
- [复现说明、审计入口与实验模板](checkpoints/2026-11-16-joint-process/REPRODUCE_RESERVOIR_JOINT_20261116.md)
- [结果图](checkpoints/2026-11-16-joint-process/reservoir_joint_figures/JointProcess.png)
- [归档元数据](checkpoints/2026-11-16-joint-process/CHECKPOINT.json) · [逐文件 SHA-256](checkpoints/2026-11-16-joint-process/MANIFEST_SHA256.json)

本地源提交为 `0666be6`。本地完整 ZIP 的 SHA-256 为 `b4c91e68715cb45d2db73f33c6dc332e058e310fd93ce6f124a8f7e0b4dea956`。
