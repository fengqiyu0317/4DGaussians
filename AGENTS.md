# 4A6000 远端连接与 GPU 验证备忘

## Windows SSH 入口

- 在本项目中连接 `4A6000` 时，优先调用 Windows OpenSSH：

  ```bash
  /mnt/c/Windows/System32/OpenSSH/ssh.exe -T 4A6000 <remote-command>
  ```

- Windows 用户 SSH 配置已定义 `4A6000` 的真实地址、端口、用户和密钥。WSL 的 `~/.ssh/config` 在 2026-09-18 尚未定义该别名；不要因为 WSL 中的 `ssh 4A6000` 解析失败就认定服务器不可达。

## NVIDIA 570/580 版本错配的隔离规避

- 2026-09-18 实测时，服务器已加载的 NVIDIA 内核模块是 `570.124.06`，而系统 `libcuda.so.1` / `libnvidia-ml.so.1` 软链接指向 `580.173.02`。普通 `nvidia-smi` 因此报 `Driver/library version mismatch`。
- 精确匹配当前内核模块的旧库仍保留在：

  ```text
  /usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06
  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06
  ```

- 在系统错配仍存在时，只为本项目的命令显式预加载匹配库；不修改系统软链接，不覆盖系统包：

  ```bash
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06
  ```

- 只读 GPU 状态查询可用：

  ```bash
  /mnt/c/Windows/System32/OpenSSH/ssh.exe -T 4A6000 \
    "env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06 \
    /usr/bin/nvidia-smi"
  ```

- PyTorch/CUDA 进程需同时预加载 `libcuda` 和 `libnvidia-ml`，并显式限定到当次查证为空闲的 GPU：

  ```bash
  env CUDA_VISIBLE_DEVICES=<confirmed-free-gpu> \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06 \
    <python-or-test-command>
  ```

- 该隔离方案已在 GPU 2 上通过 PyTorch `2.5.0+cu124` 的 CUDA 可用性、A6000 识别及单元素 CUDA tensor 实测。

## 共享服务器安全约束

- 每次 GPU 实验前都必须重新查询 GPU 进程和显存；不要把历史上空闲的 GPU 2/3 当作永久事实。
- 2026-09-18 检查时，GPU 0 有 `ylchen` 的两个长期 GR00T 服务，共占用约 13.9 GiB；GPU 2/3 当时无计算进程。该状态只是当时快照。
- 不得擅自重启服务器、重载 NVIDIA 内核模块、停止服务或终止他人进程。需要永久修复驱动时，必须先与服务器管理员及任务所有者协调维护窗口。
- 若将来系统驱动已恢复一致，普通 `nvidia-smi` 可正常工作，应停止使用上述临时 `LD_PRELOAD` 规避。
