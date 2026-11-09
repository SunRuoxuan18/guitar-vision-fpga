# Guitar Vision FPGA

基于 PYNQ-Z2 / XC7Z020 的吉他指板视觉分析与和弦识别原型。仓库保存视觉系统的核心源码、RTL/HLS 工程、测试、关键实验证据和最终实验报告，便于复核结果并继续开发。

[下载 2026-11-09 工程快照](https://github.com/SunRuoxuan18/guitar-vision-fpga/releases/download/v2026.11.09/guitar-vision-fpga-20261109.zip)

## 项目范围

- 从 Logitech C922 摄像头采集指板画面，并在 PYNQ-Z2 上运行视觉与音频服务。
- 使用端点/颜色几何和神经网络候选完成琴颈与指板跟踪。
- 估计手部 ROI 与四指尖热图，并把指尖位置投影到弦号和品号。
- 将相机画面、琴颈线、手部 ROI、指尖、弦品几何和候选音符叠加到网页。
- 结合视觉位置与音频谐波证据生成和弦候选。

项目在比赛截止前完成了端到端原型、板卡链路、模型复训和可追溯测量，但尚未达到稳定的 30 FPS 与真实演奏和弦级验收。报告明确区分真实板卡结果、回放结果、模块测试和离线指标。

## 实验报告

- [Word 报告](reports/vision_report_20261109/%E5%9F%BA%E4%BA%8EPYNQ-Z2%E7%9A%84%E5%90%89%E4%BB%96%E6%8C%87%E6%B3%95%E5%92%8C%E5%92%8C%E5%BC%A6%E8%AF%86%E5%88%AB%E8%A7%86%E8%A7%89%E5%AD%90%E7%B3%BB%E7%BB%9F%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.docx)
- [PDF 报告](reports/vision_report_20261109/%E5%9F%BA%E4%BA%8EPYNQ-Z2%E7%9A%84%E5%90%89%E4%BB%96%E6%8C%87%E6%B3%95%E5%92%8C%E5%92%8C%E5%BC%A6%E8%AF%86%E5%88%AB%E8%A7%86%E8%A7%89%E5%AD%90%E7%B3%BB%E7%BB%9F%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.pdf)

## 目录

| 路径 | 内容 |
|---|---|
| `pynq/` | PYNQ-Z2 运行端、摄像头恢复、视觉跟踪、指尖网络、音符投影和音视频融合 |
| `web/` | 浏览器可视化、交互、弦品和和弦图逻辑及前端测试 |
| `rtl/` | Verilog 图像处理、DMA、音频频谱与接口模块 |
| `hls/` | 手部卷积、ROI 投影和金字塔 LK 的 Vitis HLS 源码与 Tcl |
| `constraints/` | PYNQ-Z2 引脚与时钟约束 |
| `scripts/` | Vivado/Vitis 构建、部署、恢复和测量脚本 |
| `tests/` | Python 模块测试 |
| `sim/` | Verilog testbench 与小型测试向量 |
| `reports/` | 最终报告和关键实验记录 |
| `output/releases/fingertip_reviewed_20261030/` | 复训模型、量化候选、指标和验证回放数据 |
| `experiments/handpose_x_trial/` | handpose_x 对照源码及报告引用的对比图 |

## 复现说明

1. 安装 PYNQ-Z2 对应的 PYNQ 镜像、Python/OpenCV 依赖和 Vivado/Vitis 工具链。
2. 参照 `scripts/` 中的 Tcl 文件重建 PL 工程；不同 Vivado 版本的 IP 版本需重新锁定。
3. 将 `pynq/` 中的运行文件部署到板卡，并按实际网络地址启动视觉与网页服务。
4. 网页目录不包含 MediaPipe WASM 和浏览器模型二进制。首次运行前应按 `web/README_WEB.md` 安装或放置相应第三方依赖。
5. 前端浏览器测试依赖 Node.js 与 Playwright；可用 `PLAYWRIGHT_MODULE` 指向自定义安装位置。
6. 复训模型只用于复现实验，不等同于已在板卡全链路验证的生产模型。`pl_deployment_verified` 必须与指标文件保持一致。

## 未包含内容

为保证隐私、许可边界和仓库体积，未上传个人训练图片/录像、完整数据集、Vivado 构建缓存、BIT/HWH 大文件、第三方模型权重、比赛报名截图和含个人路径的原始诊断包。公开发布不改变第三方组件的原有许可。

## 许可

项目原创代码按根目录 [MIT License](LICENSE) 发布。第三方软件、模型、工具生成物和研究资料的许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
