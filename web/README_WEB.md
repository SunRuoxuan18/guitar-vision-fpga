# FretLens Web

## PC capture with real PL preprocessing

Open `http://127.0.0.1:8001/`, click the PC camera button and select C922.
The checked PL grayscale/edge toggle sends one 640x480 RGBA frame at a time
through the desktop proxy to PS port8084. PS rearranges RGBA to BGR, the
existing VisionPL DMA driver runs grayscale and Sobel in PL, and PS encodes
the two output planes side by side as JPEG. The response includes a unique
request token and explicit pc_camera/pl provenance. No browser image is
labelled PL on failure; the local preview is labelled separately.

The browser targets at most5 processing requests/second, keeps only one in
flight, and expires results after1second measured on the browser clock.
No comparison between PC and board monotonic clocks is made. Displayed
roundtrip includes browser pixel readback, transfer, PS/PL processing and
response decode; it is NOT optical camera-to-screen latency. Local color
preview stays independent. Raw upload is1,228,800 bytes/frame, about6.14MB/s
at5FPS, excluding response and protocol overhead.

Board code: `pynq/vision_pc_bridge.py`, using the real `pynq/vision_pl.py`.
Deployment uses `/home/xilinx/pc_pl_bridge/` and the existing optimized
vision overlay. The service template `pynq/fretlens-pc-pl.service` can be
installed for explicit starts; it is intentionally not enabled at boot.
This standalone overlay replaces audio/other PL functions; simultaneous
audio FFT and video PL processing is not established by this experiment.
Hand landmarks and local calibration still execute in the computer browser;
this bridge does not port them, automatic fret detection, or HLS LK to PL.

Evidence: `diagnostics/pc_pl_bridge_board.json` verifies known pixels against
the golden model (JPEG-loss tolerance); `diagnostics/pc_pl_browser.json`
verifies a real C922 capture, PL labels and disabling the bridge. These
short checks do not establish long-term stability or an overall speedup.

Intermediate grayscale/edge tiles are collapsed under the processing-details
control by default; hiding them does not disable PL preprocessing or tracking.
Local preview pixel loops are skipped when neither a processed main view nor
the intermediate tiles need them. Successful ROI tracks refresh spatially
distributed features every15updates (or below28points), only after forward/
backward, RANSAC and appearance checks pass with>=.85 inlier ratio. At most
48features remain; a failed refresh keeps the validated old track. This does
not reacquire an occluded/lost neck. PS LK already uses maxLevel3 and up to
30iterations per level; the HLS experiment's single-step limit does not apply
to this PS implementation.

## PC neck tracking and lattice detection

With C922 connected and PL enabled, click the neck selection button. The
image freezes; select four perimeter corners in order: sixth-string start,
sixth-string end, first-string end, first-string start. The exact frozen
frame seeds the tracker. Later images and their returned geometry share
the same stream, sequence and selection-generation identity.

PL grayscale feeds the existing PS ROI pipeline: pyramidal LK (four levels),
forward/backward consistency, RANSAC homography and appearance checks;
perspective rectification; directional Sobel projections and six-string /
equal-temperament fret-spacing validation. Two consecutive lattice results
must agree in rectified coordinates before lines are published. Full-frame
PL edge preview and rectified PS directional Sobel are distinct stages.
No Hough fallback or trained detector is claimed. Only visible frets are
measured; absolute fret numbers and complete22-fret coverage are unknown.

While tracking, the main view uses the exact processed color frame for
correct overlay alignment; this reduces its cadence to processing cadence.
Without tracking, local color preview remains independent and faster.
Lost/stale/occluded geometry is withdrawn; select again after loss. Four
bounded independent stream contexts prevent cross-tab tracker corruption;
hardware-busy responses drop frames rather than queue old images.

The experimental one-step HLS sparse LK overlay is NOT the production
pyramidal LK backend and is not loaded by this service. Integration into the
same PL design still requires hardware changes, accuracy and latency checks.
Evidence: `diagnostics/pc_flow_board.json` and `pc_flow_browser.json` use
synthetic moving guitar pixels through real PS/PL; they do not establish
real-guitar accuracy. Physical C922 transport was tested separately.

这是 `ZynqAudioLab` 的浏览器端指法与音频融合工作台，使用原生 HTML、CSS 和 JavaScript，无需安装前端框架。

## 简洁界面（2026-10-15）

当前使用地址：`http://127.0.0.1:8001/`。在 `web` 目录运行
`node serve_board.mjs` 可启动此端口的板卡代理。

- 主界面：连接摄像头 / 连接板卡音频、大视频、自动找弦与品丝 / 指尖检测，以及和弦、音高、音阶和指尖位置。
- 手动标定、起止品位和清除标定位于“标定设置”。自动模式无需点击四角；只输出有图像支持的可见线条，不保证全部 22 品或绝对品号。
- 当前自动定位是 OpenCV 几何基线，不是已训练的深度模型。合成移动图通过，公开实拍样本尚未通过。板卡检测过慢时只预览，30 秒后重试，不显示过期线条。
- 视频优化：摄像头请求 MJPEG（摄像头输出的压缩图像），保留四帧驱动缓存保障持续读取；支持时直接发送摄像头 JPEG，省去彩色解码与再次压缩。采集循环不额外等待，灰度历史降到约 10 Hz，手动跟踪开启后恢复逐帧灰度。实际协商结果在 `/api/vision/status` 的 `provenance.capture_mode`。自动定位单独在后台运行，只保留一个任务，不让网页等识别。线条与其原始图像配对，最多保留 80 ms；浏览器拒绝倒退帧。
- 板卡视觉服务启动前运行 `keep_camera_awake.py`，只关闭 C922 的 USB 运行时省电。部署服务文件时也要复制此脚本；这不代表所有 USB 驱动停采原因都已排除。
- Linux 摄像头另需部署 `camera_controls.py`：关闭 `exposure_auto_priority`，保留自动曝光并禁止因曝光降低帧率。暗处画面可能变暗，应补充照明；是否成功在 `provenance.fixed_exposure_cadence`，不要只看配置帧率。
- `capture_fps` 是最近实际发布速率，`processing_ms` 是 OpenCV 读出后到发布前的处理时间；网页 `FretVision.metrics()` 另提供取图和解码耗时。这些数值不代表镜头到屏幕的完整延迟，不能用配置的 30 fps 代替实际帧率。
- 电脑摄像头、浏览器音频、导入、模拟演示、灰度/边缘、波形、频谱与状态位于“更多输入与调试信息”。
- 以下为早期界面说明；板卡摄像头通过 PS 服务接入，电脑摄像头与音频仍可单独测试。

## 当前可以直接使用的功能

- 第一屏显示指板画面、琴弦/品格覆盖层和融合识别结果。
- “启动摄像头”会调用浏览器摄像头，优先请求 1280×720、60 fps；授权后可在下拉框切换内置摄像头或C922。
- “启动音频”会加载浏览器音频输入，可选择 Scarlett 2i2，并进行基础单音检测。
- “运行完整演示”会模拟 G Major 指法、G/B/D 三个频谱峰值、和弦与音阶候选。
- 演示结果始终标记 `DEMO`，不会伪装成真实 FPGA 或视觉模型输出。
- 波形、频谱、时间线和 PS/PL 分工保留为辅助证据，不再抢占指法识别的主位置。

## 启动方法

摄像头和麦克风需要通过 `localhost` 或 HTTPS 使用。在 PowerShell 中运行：

```powershell
cd D:\FPGA\ZynqAudioLab\web
node server.mjs
```

然后打开：

```text
http://127.0.0.1:8000/
```

同一台电脑上的 Chrome、Edge 或 Codex 内置浏览器都可以访问。关闭 `node server.mjs` 后，这个地址就无法打开。

## 必须诚实区分的三种状态

```text
DEMO
模拟指法、和弦和音阶，用于检查最终界面和数据格式。

LIVE INPUT
真实摄像头预览 + 真实浏览器音频；目前只有基础单音检测，尚未完成视觉关键点和多音融合。

FPGA / PS
当前由 PL 输出 FFT 和谱峰，由 PS/Python 输出多音与和弦结论；视觉关键点、弦品坐标与融合尚待接入。
```

## 后续接入接口

视觉适配器建议输出：

```text
{ timestamp, fingertips[], stringFretCandidates[], confidence }
```

FPGA 音频适配器建议输出：

```text
{ timestamp, onset, peaks[], notes[], level, confidence }
```

融合层按时间戳对齐最近的视频帧和音频起音，再输出：

```text
{ chord, chordNotes[], scaleCandidates[], agreement, confidence }
```

当前页面已经按这三类证据划分区域，后续接入模型和 PYNQ 数据时无需再次重做页面结构。

## 已完成的 AUD4 音频连接

板卡运行 `sudo sh pynq/start_audio_music.sh`，服务直接提供页面和同源 `/api/audio`。
电脑与板卡网络连通后打开 `http://192.168.2.99:8081/`，点击“FPGA 音频”。
这里只显示 PS 读取真实 PL 频谱后的推断。内部测试源会单独标明 A4/C/Am 测试，
静音、未知、停止更新或连接失败会清空旧结论。浏览器音频仍是独立的电脑测试功能。

本地 8000 服务可代理到板卡，默认目标 `http://192.168.2.99:8081`；
换地址时设置 `FPGA_API_URL` 后重启 `node server.mjs`。没有网线或服务未启动时
显示离线，不会自动改成模拟数据。摄像头依旧使用电脑浏览器连接的设备。
# 2026-10-16 指板与指型更新

默认“只看指板”在存在可靠几何时裁剪并加红框，丢失时回到全景。
所有手部推理与标定仍采用原图坐标。灰度/Sobel 小图在浏览器计算，最高 10 Hz；
视频显示和缩略图分别更新。它们不能当作实时 PL 视觉加速的证明。

`fingering.js` 提供 23 个开放和弦指型、E/A 横按、七种自然调式与两种五声音阶。
`fingering_panel.js` 将音频结果和视觉位置接入图示。
“本次观察”显示指尖投影，匹配当前音频 MIDI 的位置标绿，未观察的弦保持未知。
“和弦参考”显示理论指型，不能反推整只手的真实姿势。自动检测只给相对格时，
图中保留相对编号，不擅自认定琴枕与绝对品号。手机的音阶图可横向滚动。

可选 PS YOLO 使用 `vision_server.py --yolo-model ... --yolo-metadata ...`，
模型必须真正训练过 `fretboard` 类。候选框之后仍要检查图像中的弦与品丝。
无权重时不启用此路径；目前没有完成模型训练、PL 部署或新一轮上板验收。
部署与恢复见工程根目录 `RETURN_CHECKLIST.md`。

PC 摄像头的“板卡视觉处理”和“自动找弦与品丝”现在支持不选四角启动。
板卡先自动寻找六根弦和至少五条符合间距规律的可见品丝，再进入金字塔 LK
跟踪；遮挡或跟丢后清除线条并自动重新搜索。框选琴颈仅作为可选辅助。
灰度和全图 Sobel 在 PL；边缘细化、霍夫检测、光流、透视变换与区域分析在 PS。
初次搜索可能约需三秒，期间保持实时原画面，过期几何不会叠加到新图像上。
目前通过的是合成琴颈经真实板卡的测试，实琴准确率仍需现场验证。

