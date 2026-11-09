# 音频系统：从这里开始

工程根目录：`D:\FPGA\ZynqAudioLab`。音频路径包括采集、FPGA 频谱计算、ARM 多音与和弦推断、网页显示。

## 2026-10-16 当前入口

网页为 <http://127.0.0.1:8001/>。本次新增指板区域放大、红框、网页灰度/Sobel 预览，以及实际观察、参考和弦与候选音阶三类指型图。自动指板与手指位置仍需实琴验收，不能将参考图当作完整实弹结果。

断电后恢复以 `RETURN_CHECKLIST.md` 为准；下方较早的地址和恢复命令作为历史记录保留。比赛报告在 `reports/EXPERIMENT_REPORT.docx`，YOLO 准备情况见 `vision/yolo_README.md`。本次离线更新未下载到断电的板卡，音频 AUD4 镜像没有修改。

2026-10-14 算法升级及回来后的测试顺序见 `MUSIC_UPGRADE.md`：新增和弦类型、八度检查、两帧确认、连续音阶候选；FPGA 仍使用 AUD4。

## 2026-10-15 已验证音频版本

和弦显示现已升级为 `harmonic_tracking_v3`：过滤短暂跳变，对延音中的缺音短暂保持原和弦，真实换和弦需连续确认。当前使用 [简化页面](http://127.0.0.1:8001/)，刷新后点击“连接板卡音频”。页面会标明“延音保持”或“换和弦确认中”；静音、断线会清空。合成音频和板上测试通过，真实吉他需再弹奏验收。主要代码在 `pynq/music_tracking.py`；本次没有修改 FPGA 音频电路。

修正 I2S 发送端的声道标记采样时序后，用户已确认 `SW0=1、SW1=0` 吉他回放声音清晰。新版仍标识为 AUD4；请以文件哈希区分旧版：`aced9993f1beddde4a4bca3fe667405252e4c4345d4288d0078a74a15e90b610`。已封存到 `output/release_AUD4_clean_i2s_20261015`，并更新 `output/audio_music` 和板卡主目录的 BIT/HWH。

板卡当前地址 `169.254.165.100`，电脑有线地址 `169.254.165.99`，均为 `/16`。当前网页用电脑的 `http://127.0.0.1:8000/`，点击 **FPGA 音频**，代理连接板卡 8081。下面旧记录中的 `192.168.2.99` 和 JTAG 加载步骤不作为本次推荐恢复流程。

重启后，等待板卡 Linux 启动完成，通过 SSH 登录板卡，执行：

```sh
cd /home/xilinx/audio_music
sudo python3 -B pynq/manage_audio_music.py stop
sudo bash -lc 'cd /home/xilinx/audio_music; python3 -u -c "from pynq import Overlay; Overlay(\"audio_music.bit\"); print(\"AUD4_LOADED\")"'
sudo python3 -B pynq/manage_audio_music.py start
```

使用匹配的 BIT/HWH 通过 PYNQ 加载；本次 JTAG 单独加载后读 AXI 曾使 Linux 无响应，原因未完全确定。不要同时运行多个 MMIO 读取程序。此处只更新手动恢复流程，未配置开机自动加载。

本次新验证：14 项 RTL 回归、2240 个独立 I2S 串行位检查、实际 PL→PS 的 A4/C/Am 及两帧和弦确认、真实网页 10 次连续实时帧读取。内部 setup/hold 余量为 `+0.041/+0.032 ns`；外部 I2S 时序约束及完整 CDC 签核仍待完善。真实吉他音符和和弦准确率尚未验收，下一步先逐根空弦再测 C/Am/G。

## 1. 整条链路在做什么

```text
电吉他 → Scarlett 的乐器输入 → Scarlett 耳机输出 → 板卡 LINE IN
→ 板载 Codec 把模拟电压转成数字音频
→ I2S 接收 → PCM 样本 → 双 BRAM 帧缓存 → Hann 加窗
→ PL 内 AMD 4096 点 FFT → 功率频谱与 24 个局部峰
→ AXI-Lite 寄存器 → PS 上 Python/NumPy
→ 谐波模板拟合 → 多个音符 → 和弦名称 → HTTP API → 网页
```

- **Codec**：音频编解码芯片，这里负责模拟与数字转换，以及耳机输出。
- **PCM（Pulse Code Modulation，脉冲编码调制）**：按时间顺序排列的音频幅值数字。本工程分析输入为有符号 16 位。
- **I2S（Inter-IC Sound，芯片间音频总线）**：传 PCM 的线路；有位时钟 BCLK、左右声道标记 LRCLK、数据线。
- **I2C（Inter-Integrated Circuit，芯片间总线）**：给 Codec 配置寄存器，与传音频的 I2S 分工不同。
- **BRAM（Block RAM，块存储器）**：FPGA 内的专用存储块，先保存一帧音频。
- **Hann 窗**：把一帧两端平滑压低，减少截断造成的频谱泄漏。
- **FFT（Fast Fourier Transform，快速傅里叶变换）**：从时间幅值序列得到频率成分，不是失真或延迟效果器。
- **PL（Programmable Logic，可编程逻辑）**：运行 Verilog 硬件与 FFT IP。
- **PS（Processing System，处理系统）**：板卡 ARM 处理器运行 Linux、Python 和网页服务。
- **AXI-Lite**：ARM 读写 FPGA 寄存器的总线。
- **MMIO（Memory-Mapped Input/Output，内存映射输入输出）**：把硬件寄存器当成地址来读写；本机部署使用有权限的 `/dev/mem` 32 位对齐访问。
- **API（Application Programming Interface，应用程序接口）**：网页访问 `/api/audio`，得到音符、和弦、频谱和运行状态的 JSON 数据。

耳机回放和 FFT 分析是两条支路。PS 暂停频谱更新以读取一帧时，不会主动暂停耳机音频。

## 2. 代码在哪里，每个模块学什么

| 文件 | 作用 | 最重要的知识点 |
| --- | --- | --- |
| `rtl/pynq_audio_board_top.v` | 板卡音频连接、初始化、开关与 LED | 顶层实例化、引脚映射、状态诊断 |
| `rtl/pynq_audio_stream_top.v` | 接收、处理、回放数字音频 | 模块之间的 sample/valid 连接 |
| `rtl/adau1761_init.v` | 配置板载 Codec | I2C 状态机与 ACK 应答 |
| `rtl/audio_i2s_rx.v`、`rtl/audio_i2s_tx.v` | I2S 音频收发 | 串并转换、声道、移位时刻 |
| `rtl/audio_sample_cdc.v` | 音频样本跨时钟域 | 保持数据、同步控制脉冲 |
| `rtl/audio_fft_capture.v` | 两个 4096×16 位采样缓存 | 同步 RAM 读、读写所有权、溢出恢复 |
| `rtl/audio_fft_engine.v` | 读帧、乘窗、驱动 FFT、计算功率 | 状态机、AXI-Stream valid/ready 握手、定点数 |
| `rtl/hann4096.mem` | 4096 个 Q15 窗系数 | ROM、定点比例与乘法移位 |
| AMD `audio_fft_4096` IP | 真正计算 FFT | 使用成熟 IP、配置位宽、输出复数 |
| `rtl/audio_spectrum_axi.v` | 保存 2048 个功率值、扫描谱峰、供 ARM 访问 | AXI-Lite、冻结快照、帧序号 |
| `rtl/audio_analysis_test_source.v` | 内部 A4/C/Am 测试源 | 可重复的板上验收；只影响分析输入 |
| `rtl/audio_music_pl.v` | 连接音频、FFT、AXI 模块 | 硬件层次与接口连接 |
| `pynq/audio_fft.py` | ARM 安全读取同一帧频谱 | 32 位对齐 MMIO、冻结后等待、读完释放 |
| `pynq/audio_music.py` | 识别多音与和弦 | 谐波模板、非负联合拟合、音级集合 |
| `pynq/audio_music_server.py` | 采样线程与 HTTP 服务 | Python 线程、JSON、接口状态 |
| `web/index.html`、`styles.css`、`app.js` | 页面结构、样式、数据交互 | HTML/CSS/JavaScript、fetch、Canvas |

`sim/` 是硬件 testbench，`pynq/tests/` 是软件测试，`scripts/` 是构建、下载、串口部署和取回证据的脚本。Vivado 工程在 `diagnostics/audio_music_build/audio_music.xpr`；最终 bitstream 在 `output/audio_music/audio_music.bit`，描述文件为同名 `.hwh`。

## 3. 为什么频谱峰不能直接当成几个音符

一根弦弹 A3，除了 220 Hz，通常还会出现 440、660、880 Hz 等谐波。把最大的几个峰直接标成音符，会把一根弦误认为多个音。

本程序同时比较一个音的基频和谐波，再把多个候选共同拟合到频谱。识别出 C、E、G 三种音级后，可以推断 C 大三和弦。有低音转位时可输出 `C/E`；无可靠结论时输出 `unknown`。分数是模型评分，不是经过真实吉他统计校准的概率。

支持的名称包括大小三和弦、减/增和弦、常见七和弦、sus 和 power chord。识别一个和弦并不等于确定整首曲子的调；可选音阶结果只表示当前音级与某音阶相容。

## 4. 你回来后的操作

### 板卡没有断电

最终 AUD4 已通过 JTAG 下载时，保持板卡供电即可。吉他测试用 `SW0=1、SW1=0`，沿用之前确认正常的音频线与增益。PS 网页服务运行在 8081 端口；服务状态见本次测试记录。

开关用途保持已验证音频版本的定义：

| SW0 | SW1 | 耳机音频 |
| --- | --- | --- |
| 0 | 0 | FPGA 内部测试音 |
| 0 | 1 | 原始串行直通诊断 |
| 1 | 0 | 解码后回放吉他，先使用这一档 |
| 1 | 1 | 完整音频处理路径 |

`SW0=1、SW1=0` 时，LD0 表示每声道检测到 32 个 BCLK，LD1 表示 16 个，LD2 表示其他数量；这三盏灯此时不是音量灯。LD3 表示 Codec 初始化错误。`SW0=0、SW1=0` 时，LD0 表示音频时钟锁定，LD1 表示初始化完成，LD2/LD3 记忆时钟异常/按键复位。灯号必须结合档位解释。

网线暂时没接也能通过 USB 串口部署和检查；浏览器直接访问板卡仍需要建立电脑与板卡的网络连接。

接好网线后，板卡目前地址为 `192.168.2.99/24`。电脑有线网卡需位于同一网段，例如 `192.168.2.10`、掩码 `255.255.255.0`，不必为了两者互通填写网关。不要修改 Wi-Fi 网卡。

打开 `http://192.168.2.99:8081/`，点击 **FPGA 音频**。页面中的频谱来自 PL，音符与和弦来自 PS。没有声音时应显示静音或未知，不应保留上一个和弦。不要用“运行完整演示”作为硬件成功证据。

电脑上的 `http://127.0.0.1:8000/` 也已保留，使用同样的按钮访问代理接口。它与板卡网页有区别：摄像头插在电脑时，浏览器摄像头功能可在 localhost 下使用；普通 HTTP 的板卡地址可能受浏览器摄像头权限限制，后续摄像头阶段再配置。

### 板卡断电后重新开机

`.bit` 配置会随断电丢失。SD 卡保存 Python 文件，不代表自动加载了这一版 FPGA。

1. 保留 SD 卡，连接板卡 PROG/JTAG USB，开机并等待 PYNQ Linux 启动。
2. 电脑 PowerShell 执行：

```powershell
cd D:\FPGA\ZynqAudioLab
& .\scripts\program_audio_music.ps1
```

脚本确认并设置 PS FCLK0=125 MHz、下载 AUD4、读回版本 `41554434`。它不会完整重置 PS 或 DDR。单独在 Vivado 点 Program Device 不替代必要的时钟步骤。

3. 板卡终端执行：

```sh
cd /home/xilinx/audio_music
sudo sh pynq/start_audio_music.sh
```

输入板卡 sudo 密码。服务第一次准备模板可能需要几秒到十几秒，网页这时显示等待。不要同时启动两个访问频谱的程序；运行板上 smoke test 前先停止服务。

## 5. 仿真与验收

本次已完成：12 组音频硬件回归；FFT 四组共 8192 个数值检查；AXI 寄存器 25061 项检查；电脑和板卡 ARM 上 43 项软件测试；最终板上 A4/C/Am 识别；网页 11 项适配器测试和三种屏幕尺寸验证。最终下载读回版本 `41554434`、DONE=1、CRC错误=0。

最终内部建立时间余量 +0.011 ns、保持时间余量 +0.025 ns，DRC 无错误；外部接口和 CDC 未完整签核，见下方边界。最终封存目录为 `output/release_AUD4_20261014`，其中 `SHA256SUMS.txt` 可核对文件版本。

电脑运行硬件回归：

```powershell
cd D:\FPGA\ZynqAudioLab
& .\scripts\run_audio_regression.ps1
& D:\AMD\2026.1\Vivado\bin\vivado.bat -mode batch -source scripts/test_audio_music.tcl
```

FFT testbench 用独立 NumPy 参考检查静音、A4、C、Am 四组完整正频谱，每组 2048 个值，包含未知值检测。不是只看波形似乎像一个正弦。

板卡上验证整个 PL→PS 链路：

```sh
cd /home/xilinx/audio_music
sudo env OPENBLAS_NUM_THREADS=1 python3 -B pynq/test_board_music.py --dev-mem --output board_music_smoke.json
```

应通过 A4、C4/E4/G4、A3/C4/E4 三组测试，并恢复分析模式 0。内部测试信号不会变成耳机里的测试和弦。测试输出不是实际弹琴识别准确率。

重新完整构建：

```powershell
& D:\AMD\2026.1\Vivado\bin\vivado.bat -mode batch -source scripts/build_audio_music.tcl
```

只重建现有工程可加 `-tclargs --reuse`。脚本执行物理优化，并在最终时序失败时拒绝生成新 bitstream。

## 6. 已知边界与下一步

- 48 kHz 的 4096 点窗口是 **85.33 ms**，频率间隔是 **11.71875 Hz**；这不是 1 ms 检测系统。
- 真实识别还包括 ARM 计算、网页刷新和拨弦稳定时间。实测耗时见 `diagnostics/audio_board_evidence/board_music_benchmark.json`，不能用 FPGA 时钟频率代替系统延迟。
- 目前 ARM 上 C 和弦推断平均约 **463 ms**，首次建立模板约 **12.6 秒**。这是可运行的基础识别版，快速乐句仍需后续优化；其他复杂和弦的耗时可能更长。
- 当前非重叠采样窗用于基础版本；快速乐句后续可增加重叠窗、起音检测和候选音限制。
- 低音基频分辨率有限，靠谐波补充；失真、重叠谐波、弦噪、弱音、完全八度重合可能造成错误。
- 用户先前确认的 AUD3 音频版本已冻结在 `output/verified_audio_AUD3_20261014`，需要时按其中 `RESTORE.md` 恢复。
- 内部时序通过不等于 Codec 引脚时序与所有跨时钟域已完成签核；现有外部 I2S 延时约束与 CDC 检查仍有待完善，报告保留在 `output/audio_music/`。
- 回来后先用干净吉他声验证六根空弦和 C/Am/G，记录误判；再开始摄像头的指板标定、指尖定位与音频候选融合。

过程记录：`AUDIO_MUSIC_WORKLOG.md`。寄存器说明：`SPECTRUM_REGISTER_MAP.md`。PS 算法说明：`pynq/README_MUSIC.md`。

后台服务也可以在板卡上管理：

```sh
cd /home/xilinx/audio_music
sudo python3 -B pynq/manage_audio_music.py status
sudo python3 -B pynq/manage_audio_music.py stop
sudo python3 -B pynq/manage_audio_music.py start
```

`start` 默认选普通音频分析模式。`--test-mode 2` 仅用于内部 C 和弦验收，界面会标记内部测试。不要在服务运行时直接操作同一组 FFT 控制寄存器。
