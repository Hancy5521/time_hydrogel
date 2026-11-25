# AAm/MBAA 单体成网与过程逆设计审阅包

2026 年 11 月 25 日。本包对应单体到成网及双目标过程设计的最新计算结果。请先将 ZIP 全部解压，再打开 [READ_FIRST.html](READ_FIRST.html)。页面无需加载外部资源。

## 先看结论与证据

48 张新规则卡、每张 16 个配对地址，共有 768 个配对起点。五个分支用相同随机地址从未接键单体重新开始，共 3,840 条轨迹，数值时间步为 0.0125。配对增益区间按规则卡计算。

| 目标 | 冻结静态编辑 | 冻结分时策略 | 达标比例差及 95% 区间（百分点） |
| --- | ---: | ---: | ---: |
| 快速成网 | 365/768 | 445/768 | +10.42 [5.60, 15.10] |
| 推迟成网、后段追平 | 61/768 | 267/768 | +26.82 [23.70, 29.95] |
| 同一起点的两条分支各自达标 | 34/768 | 176/768 | +18.49 [14.97, 21.88] |

快速方案提前投料并提高引发和增长。延迟方案保留早期链增长，随后提升第二位点的反应活性。最终配对的两套策略由先前群体模拟选定。独立程序重放了 240 份原生事件日志和 21,243 个事件。 [最终报告](run/final_paired_process/RESULT_ZH.md) · [冻结](run/final_paired_process/FROZEN.json) · [原始确认数组](run/final_paired_process/RAW.npz) · [统计](run/final_paired_process/SCORED.json) · [独立审计](run/final_paired_process/INDEPENDENT_AUDIT.json)

## 按研究问题审阅

| 问题 | 程序及记录 |
| --- | --- |
| 单体、链和网络怎样随时间演化 | [反应适配器](run/reaction_adapter.py) · [主协议](run/PROTOCOL_ZH.md) · [事件与网络审计](run/AUDIT_RAW.json) |
| 局部观测能否反推四维材料规则 | [编码方法](run/ENCODER_METHOD_ZH.md) · [代码](run/code_encoder.py) · [同预算后移观测](run/continuous_bridge_transfer/RESULT_ZH.md) |
| 按时间目标反选方案后，能否独立重启确认 | [原冻结方案](run/FROZEN.json) · [分时投料](run/scheduled_edit/RESULTS_ZH.md) · [第二位点控制](run/pendant_timing/RESULTS_ZH.md) · [最终配对](run/final_paired_process/RESULT_ZH.md) |
| 步长变化、隐藏空间及功能读出有什么影响 | [步长确认](run/resolution/SCORED.json) · [空间补全](run/spatial_completion_retry/INDEPENDENT_AUDIT_ZH.md) · [结构传输](run/transport_proxy/RESULT_ZH.md) |
| 公开实验可验证到哪一环 | [外部数据逐项核查](run/EXTERNAL_EVIDENCE_ZH.md) |
| 全部发现、失败及可复现入口 | [主报告](run/RESULT_ZH.md) · [历史复现说明](run/REPRO_ZH.md) |

训练中将引发、链增长、悬挂位点活性和桥连偏置表示为同一材料的四个固定坐标。工艺变化和材料坐标分开施加。后移同样三帧降低了桥连码识别误差，固定零偏置仍是竞争力较强的基线，个体编码尚无逆向选择增益。空间验证使用体素计数，隐藏区域未超过体积分配基线。传输验证使用单位边网络。对应真实材料配方、导电、力学和第二位点分时可制造性的证据仍需同母批实验。 [完整证据边界](run/RESULT_ZH.md)

## 解压后离线核对

使用 Python 3.11 或以上版本，从本包根目录运行。

~~~text
python -B verify_manifest.py
python -B verify_review.py
python -B replay_events.py
~~~

第一条只用标准库逐文件验证 SHA256。第二条从留存数组复算达标数，第三条重放留存的最终事件。评分复算与事件重放应使用已具备相应依赖的环境，可参考 [requirements.txt](requirements.txt)。这三个入口都保留冻结记录。若要生成新轨迹，先阅读 [历史复现说明](run/REPRO_ZH.md)，在另一个目录进行登记。

- run/ 包含原始代码、登记、冻结预测、合成轨迹、图和各阶段审计。
- native/ 包含原始解码器和反应场的代码依赖。
- provenance/ 包含运行收据。原收据的路径及哈希保留作为来源记录，以上根目录命令能够使用解压后的相对目录。
- [MANIFEST.sha256](MANIFEST.sha256) 用于核对包内文件完整性。
## 公开复核镜像

本目录包含冻结代码、合成模拟数组、图件和评分。合成数组与图件保留了原始字节。机器路径已替换，三个源码入口改为使用随包依赖，复核脚本在检查镜像文件哈希后读取对应的原始收据哈希。镜像与原始封存压缩包分别校验。

- 原始封存 ZIP 的 SHA-256 为 **e496f8c6e7c0b297d4b20e957bad0047258cfafc780fb1b2e6e60b32041a4c7a**。
- [逐文件变更与原始和镜像 SHA-256](PUBLIC_TRANSFORMS.json)
- [本镜像的 SHA-256 清单](MANIFEST.sha256)
- [公开镜像的内容范围](provenance/PACKAGE_SCOPE.json)

从此目录执行：

~~~text
python -B verify_manifest.py
python -B verify_review.py
python -B replay_events.py
~~~

根目录的清单脚本需要 Python 3.11+。评分需要 NumPy，事件回放还使用随包提供的源码依赖，见 [运行依赖](requirements.txt)。历史源码与收据保留原有实验配置，编辑的路径以本镜像根目录为相对参照。成网与逆设计结果来自已封存的模拟，目标材料的独立实物证据见[外部证据说明](run/EXTERNAL_EVIDENCE_ZH.md)。
