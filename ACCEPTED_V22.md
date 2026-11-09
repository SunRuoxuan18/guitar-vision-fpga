# 最新验收版本：v22（2026-10-23）

用户已确认：目前效果优于此前各版，保存为最新基线。后续试验不得覆盖此快照。

## 保存位置

- 完整发布目录：`output/releases/fretlens_v22_20261023/`
- 最新版本指针：`output/latest.json`
- `board_runtime.tar.gz`：从当前板卡直接保存的静态程序、模型、ARM 原生库、bit/hwh 和 systemd 配置。
- `pl/`：与板卡 SHA256 一致的 PL 文件及实现报告，来源为 `output/hand_neural_dw_wide/`。
- `source/`：当前网页、PS 程序、RTL、HLS 源文件和部署脚本的快照。
- `neck_hand_v22_20261023.tar.gz`：原始 v22 增量安装包。
- `evidence/`：已有测试报告及回放测量。`SHA256SUMS.json` 覆盖发布目录的全部有效载荷。

注意：`output/av_system/` 中存在旧 bit/hwh，不是本次验收版本；恢复时使用本发布目录。

## 当前分工与改进

电脑采集、传输并显示视频；识别仍在板卡 PS/PL。PS 优先使用当前帧手部证据，放宽快速位移关联，合并原生图像操作，减少缩放及预览转换开销，对失败的弦品搜索退避。PL 保留当前灰度、边缘、金字塔 LK 及音频设计；本次 v22 没有更换 bitstream。

同一 45 帧回放：响应率 4.63→6.03 Hz，中位往返耗时 201.67→149.72 ms，板卡处理 167.94→113.42 ms。这是既有回放证据，不代表 30 FPS 或完整和弦识别准确率。详见 `evidence/NECK_HAND_V22_RESULT.md`。

## 恢复说明

先校验 `SHA256SUMS.json`；不要将整个目录直接覆盖到正在运行的系统。

只恢复 v22 算法：在相同系统及相同 PL 基线上，将增量包解压到板卡独立暂存目录，进入该目录，以 root 执行其中的 `install_neck_model.sh`。安装器校验文件、编译 ARM 库并保存旧版本；执行时会重启应用服务。

恢复整套应用：先将 `board_runtime.tar.gz` 解压到独立暂存目录。其 `av_system/` 是实际板卡静态文件，`etc/systemd/system/` 是持久服务配置，`run/systemd/system/` 是临时运行配置，仅供检查，不应自动持久化诊断开关。保存待恢复板卡原目录与服务配置后，停止 `fretlens-av.service`，部署静态文件及已审核的持久配置，再 daemon-reload 并启动服务。此操作需要兼容的 PYNQ/Linux、Python/OpenCV 和系统库。

网页恢复使用 `source/web/` 与配套启动脚本；地址和环境变量以当前主机设置为准。`source/` 保存的是开发源码，个别文件可能包含尚未部署的实验代码；还原板卡运行效果以 `board_runtime.tar.gz` 为准。

本快照不包含操作系统镜像、Python 虚拟环境、浏览器状态或实时音视频缓存。此次保存不重启服务、不切换 PL。
