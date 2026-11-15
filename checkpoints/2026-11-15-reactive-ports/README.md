# 双口扩散与不可逆成胶

同一累计剂量在不同入口分时供给。用两个位置的暂态读数选择工艺，检验能否控制两区的成胶顺序，同时满足终态转化要求。

- [研究结果与适用范围](GEL_REACTIVE_PORT_DESIGN_ZH.md)
- [执行前登记](runs/gel_reactive_ports_20261115/EXEC_REGISTERED.json)
- [冻结结果](runs/gel_reactive_ports_20261115/SCORED.json)
- [独立求解与数值核查](runs/gel_reactive_ports_20261115/INDEPENDENT_AUDIT.json)
- [图](runs/gel_reactive_ports_20261115/REACTIVE_DESIGN.png)
- [归档文件及 SHA-256](MANIFEST.json)

该快照记录反应–扩散模拟及合成配置，源论文与公开实验链接见报告。原始本地评审包对应 SHA-256 `5bc5e6a133f097689f95383707818be75fd08d032ba3df7cdc0b8887cc2c18fc`。本目录保留已登记代码与原始结果的字节。报告移除了指向前一阶段本地文档的链接，公开论文引用仍可访问。

## 复现

在本目录使用 Python 3.13.5。原始结果位于运行目录，新结果写入 `reproduction/`。

```powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$run = 'runs/gel_reactive_ports_20261115'
New-Item -ItemType Directory -Path reproduction -Force | Out-Null
python -X utf8 -B audit_gel_reactive_ports.py "$run/EXEC_REGISTERED.json"
python -X utf8 -B gel_reactive_port_study.py "$run/EXEC_REGISTERED.json" "reproduction/SCORED.json"
python -X utf8 -B gel_reactive_port_diagnostics.py "$run/EXEC_REGISTERED.json" "$run/SCORED.json" "reproduction/DIAGNOSTICS.json"
python -X utf8 -B plot_reactive_port_design.py "$run/EXEC_REGISTERED.json" "$run/SCORED.json" "$run/DIAGNOSTICS.json" "reproduction/REACTIVE_DESIGN.png"
```

首次写出失败的状态与原执行源码保存在运行目录，最终结果见 `SCORED.json`。
