# handpose_x 替换试验记录

## 目的

验证 `handpose_x` 是否能替代当前 MediaPipe 21 点手部模型，改善吉他按弦手指尖准确率。

## 结论

不接入生产链路，不部署到板卡。

在这批真实吉他画面中，`handpose_x` 虽然能稳定输出 21 个二维点，但多个点会塌缩到一起、落到手外或偏离关节。按掌心方向旋转到手部竖直后也没有改善。当前 MediaPipe 模型虽然在遮挡和贴边场景仍会失败，但它至少提供姿态置信度，可以拒绝明显错误的结果；`handpose_x` 这个移植只返回 42 个坐标，没有可用于拒绝错误结果的置信度。

## 测试条件

- 数据：`datasets/fingertip_boost_20261030/images`
- 帧：`v01_0001`、`v01_0112`、`v01_0182`、`v01_0223`、`v01_0233`
- 关键点权重：`weights/resnet50.param`、`weights/resnet50.bin`
- 关键点权重长度：`47,188,480` 字节
- 手部检测权重：`weights/handpose.param`、`weights/handpose.bin`
- 桌面 CPU：NCNN，4 线程
- 板卡数据：未测试

## 结果

- 官方两阶段流程：5 帧各检出 1 只手。
- 检测耗时：`14.0-19.8 ms/帧`
- handpose_x 耗时：`24.7-33.6 ms/手`
- 固定手框直接推理：关键点大量落到手外，不可用。
- 官方 YOLO + ROI 扩展：能检出，但指尖和关节仍明显偏移。
- 原版正方形 `1.1x` 裁框：没有解决精度问题。
- 按掌心方向旋转到竖直：没有解决精度问题。
- MediaPipe 诊断对照：8 个候选手框接受 7 个。该轮把阈值降到 `0.2` 仅用于观察骨架，不代表生产阈值下的准确率。

## 输出

- `output/comparison_sheet.jpg`：固定手框直接推理。
- `output/two_stage_sheet.jpg`：官方 YOLO + handpose_x 两阶段流程。
- `output/mediapipe_sheet.jpg`：同画面 MediaPipe 基线。
- `output/square_roi_comparison_sheet.jpg`：原版正方形裁框对照。
- `output/three_way_comparison_sheet.jpg`：原方向、竖直方向、MediaPipe 三列对照。

## 复现

```powershell
uv run --python 3.12 --with ncnn --with pillow --with numpy `
  python experiments/handpose_x_trial/run_detector_trial.py
```

同手框 MediaPipe 对照需要 OpenCV。当前 Windows 环境使用已缓存的
`opencv-python-headless 4.10.0.84`，脚本通过 `PYTHONPATH` 指向对应缓存目录。

## 后续建议

继续使用现有 21 点模型和已标注数据训练，优先处理手部裁框、遮挡、低置信度拒绝和指尖局部跟踪。若仍要采用 handpose_x，需要重新训练或微调并补充置信度头，同时解决 ARMv7 板卡上的 NCNN 编译、量化与内存占用问题，不能作为直接替换。
