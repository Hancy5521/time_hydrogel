# 冻结仿真源码

本目录按原工作树的相对层级保存 23 个逐字节核对过的 Python 文件。其中 21 个文件匹配本次研究登记中的 245 条源码哈希引用，另有 2 个文件提供本地网络性质计算和图线性响应。映射及哈希见 [NATIVE_SOURCES.json](NATIVE_SOURCES.json)。

这些文件覆盖本包的单体起步模拟、四坐标规则解释、切换工艺和原生事件的独立重放。研究入口在公开镜像根目录。历史登记中的机器路径已替换为从镜像根目录开始的相对路径，原始哈希可查阅镜像根目录的逐文件变更账本。

随包提供的 experiments/aam_gene_field/audit.py 在当前审计中提供已有的事件图实现。它另有历史研究的单独 qualify 命令，依赖更早试验的数据。该旧命令的输入不属于此次 AAm 验证包。

如需在解压目录导入原生模拟，请先安装评审包 requirements.txt 中的依赖，再从包根目录指定下列环境变量。

    $env:AAM_REACTION_WORKTREE = (Resolve-Path snapshot/native).Path
    $env:PYTHONPATH = (Resolve-Path snapshot/run).Path
    $env:NUMBA_CACHE_DIR = Join-Path $env:TEMP 'aam_review_jit'

短且独立的 JIT 路径可避免不同进程复用旧缓存。Numba 编译时产生的缓存不属于源码与审计证据。
