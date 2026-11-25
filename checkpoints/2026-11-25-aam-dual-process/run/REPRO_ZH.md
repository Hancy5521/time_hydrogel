# 可复现入口

本目录是 2026-11-25 单体到成网的 AAm/MBAA 粗粒化计算研究。主报告入口为 [RESULT_ZH.md](RESULT_ZH.md)。所有核心源码保留历史基础场，引入的文件只在本运行目录。数据均为合成轨迹或其独立评分，公开文献及实测范围在 [EXTERNAL_EVIDENCE_ZH.md](EXTERNAL_EVIDENCE_ZH.md) 区分。

## 验证已有输出

- [REGISTERED.json](REGISTERED.json) 绑定源码、时间、局部字段、作用坐标和实验预算。
- [ACQUISITION.json](ACQUISITION.json) 绑定 A/B 开发及盲测局部读数、C 零码锚点。确认用的码另存。
- [FROZEN.json](FROZEN.json) 和 [POOLED_NOMINAL.json](POOLED_NOMINAL.json) 在读取全菜单的新 C 真值前同时锁定。两个评分入口为 [score_loop.py](score_loop.py) 与 [audit_scores.py](audit_scores.py)。读取既有输出时可从 [INDEPENDENT_SCORE_AUDIT.json](INDEPENDENT_SCORE_AUDIT.json) 核对全部方法。
- [AUDIT_RAW.json](AUDIT_RAW.json) 对 [CONFIRMED.npz](CONFIRMED.npz) 中保留的事件日志做完整反演。审核只用已保存的数据及初始引发预算的独立重建。图形摘要见 [CLOSED_LOOP_FIGURE.png](CLOSED_LOOP_FIGURE.png)。
- 附属试验都在各自目录登记和冻结。数值步长见 [resolution](resolution/SCORED.json)，观测时间见 [topology_witness](topology_witness/RESULT_ZH.md) 和 [continuous_bridge_transfer](continuous_bridge_transfer/RESULT_ZH.md)。三维场试验见 [spatial_completion_retry](spatial_completion_retry/INDEPENDENT_AUDIT_ZH.md)，图结构传输见 [transport_proxy](transport_proxy/RESULT_ZH.md)，两种工艺机制见 [scheduled_edit](scheduled_edit/RESULTS_ZH.md) 与 [pendant_timing](pendant_timing/RESULTS_ZH.md)。

## 启动新复现实验

使用带 NumPy、SciPy、Numba 的 Python。先保留当前运行及已有冻结，复制源码到新的时间戳目录，依据 REGISTERED.json 核对基础场各文件的对应 SHA256。环境变量 AAM_REACTION_WORKTREE 指向当前原始工作树。Windows 的 NUMBA_CACHE_DIR 宜设为本次独有且简短的路径，避免跨进程已有缓存反序列化和文件名长度问题。主入口为 loop.py。

1. loop.py register 生成排他写入的登记。
2. loop.py collect 生成角色固定的开发、确认局部和 C 零码数据。
3. loop.py freeze 与 score_loop.py pooled-freeze 先保存原预测与全部选案。
4. loop.py confirm --freeze-sha256 <此次实际预测哈希> 获取不同随机地址下的 C 原始真值。
5. score_loop.py score 写评分。以原生事件图和独立程序核对。

主随机地址由目录的实际 UTC 标识与角色拼接生成。新的目录自动使用新种子。要从原样本重放结果，使用已保存的 NPZ 和冻结、登记及种子地址核对，保留现存收据。

最终双目标配对证据见 [冻结文件](final_paired_process/FROZEN.json)、[评分](final_paired_process/SCORED.json)、[独立审计](final_paired_process/INDEPENDENT_AUDIT.json)。该队列采用较细的 0.0125 模拟网格和新随机地址，两个控制方案沿用此前冻结。
