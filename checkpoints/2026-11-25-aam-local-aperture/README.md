# 水凝胶局部时空观测与成网规则反推

这是 2026 年 11 月 25 日的公开复核镜像。研究从局部时空见证或可采集的成键事件反推 AAm/MBAA 仿真中的拥挤度依赖规则，再在留出工艺中比较独立成网轨迹。结果来自合成仿真，观测设备及目标水凝胶的实物验证仍有明确条件。

- [研究报告与阶段结果](REPORT_ZH.md)
- [固定快照协议](PROTOCOL_ZH.md)、[移动视野协议](MOBILE_PROTOCOL_ZH.md)
- [冻结工艺轨迹复验协议](REPLICATION_PROTOCOL_ZH.md)、[连续规则验证协议](CONTINUUM_PROTOCOL_ZH.md)
- [简化观测校准与留出验证](CAL_PROXY_PROTOCOL_ZH.md)
- [原生仿真源码及范围](snapshot/native/README_ZH.md)

用 Python 3.13 与 [已测试的依赖版本](requirements.txt) 在镜像目录运行：

~~~text
python -B verify_manifest.py
python -B verify_review.py
~~~

第一条命令只需要 Python 标准库。第二条命令先核验清单，再从冻结数组独立复算 14 项检查，不训练模型、不重新生成轨迹、不写入原始审计收据。对个别结果可指定检查名，例如 `python -B verify_review.py replication-c continuum-c calibrated-proxy`。

本镜像保留了本研究全部 44 个冻结 NPZ 文件，字节与原归档一致。历史 JIT 缓存未包含。研究登记中的盘符和机器路径换成了镜像内路径，种子寻址继续使用冻结时的轮次标识。历史设置和早先轮次的整数随机地址用于来源校验与地址重叠检查。[逐文件变更、原始哈希与发布哈希](PUBLIC_TRANSFORMS.json) · [镜像字节校验清单](MANIFEST.sha256)
