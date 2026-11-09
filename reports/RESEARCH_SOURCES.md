# Verified Research Sources

Research scope: previous AMD/Xilinx FPGA competition reports, PYNQ-Z2 PS/PL and
YOLO feasibility, FINN/Vitis AI constraints, and licensed music references.
Reporting date: 2026-10-16, as supplied by the workspace. HTTP verification began
at 2026-10-15 16:30:59 UTC according to the available clock. This is a bounded
research snapshot, not a certification of competition eligibility or hardware.

Only this file was authored. Sources were fetched with Node/Python HTTP clients;
PDFs were parsed from memory with pypdf. No executables, models, bitstreams, or
dependencies were downloaded or installed. The board was off and was not
contacted. HTTP 200 below means content was fetched and inspected, not merely
that a URL looked plausible. GitHub branch heads were resolved through fetched
repository pages; important GitHub evidence is pinned to those revisions.

## 1. Current Requirements: Recovered Screenshots

These are user-provided local evidence. Their publisher, event edition, date,
and current submission status cannot be independently established from the
cropped images. They take precedence over unrelated historical track rules for
the scope of this research.

| Evidence | Directly visible facts | Boundary |
| --- | --- | --- |
| [competition_requirements.jpg](sources/competition_requirements.jpg), slide 19 | `赛道三 · 自主选题`; `分组、选题与实现方式`; initial group is undergraduate years 1-3; AMD FPGA/Zynq devices; emphasis on `逻辑设计、状态机、时序约束、接口协议等基本能力` | This slide contains no current deadline, judging weights, qualification result, or mandatory report template. |
| Same slide, `选题与实现方式` | Application unrestricted; RTL/HLS/AI Engine/PS software allowed; value can be acceleration, system integration, interface design, or application innovation; Vivado/Vitis 2026.1 recommended or 2025.2; must run on board and be open source under MIT/Apache-2.0 | Listing AI Engine among event-wide methods does not establish AI Engine hardware on PYNQ-Z2. |
| Same slide, `初级组推荐方向：PYNQ` | PYNQ recommended, not required; PS Python/Jupyter organizes algorithms/data; PL HLS or RTL implements computation packaged as an overlay | Does not require YOLO, FINN, a DPU, or HLS instead of RTL. |
| [submitted_project.png](sources/submitted_project.png) | Team number 42573; AMD; `AMD-自主选题（初级）`; title `FPGA信号采集及协议分析仪`; platform includes AMD 7 Series FPGA/Zynq-7000/UltraScale/UltraScale+ | An application form image does not prove final acceptance, submission success, advancement, or an award. |
| Visible application description | Simultaneous multichannel acquisition, edge triggering, buffering, timing analysis; possible UART/SPI expansion; FPGA sampling/trigger/storage/protocol preprocessing; PYNQ or Linux/Python configuration, waveform display, and export | The visible description mentions UART/SPI as expansion. It does not explicitly name I2S/I2C. Using I2S/I2C to substantiate the title is the current project direction, not a quotation from the form. |

Screenshot SHA-256 values:

- `competition_requirements.jpg`: `6830e936e9f39e9d11a1a5123f2d451bc4d3cc363094f99ed664222348ebe260`.
- `submitted_project.png`: `26b1d6afe3f2dc743c26462b06b60c5817447461c3c4171a9acc71f7aba9fb7e`.

**Consequence for the report:** preserve the registered title. Present I2S signal
capture/decoding and I2C codec-control transaction evidence as the instrument's
core. Music analysis and camera/YOLO results can illustrate an application.
Object detection alone cannot substantiate acquisition, triggering, protocol
decoding, or timing analysis. Driving codec configuration is also not equivalent
to implementing a general passive I2C analyzer.

## 2. Historical Report Files and Exact Structures

### 2.1 Historical task-upstream template: ten sections

- **Fetched HTTP 200:** [2025 task README](https://gitee.com/Vickyiii/hlstrack2025/raw/master/README.md).
- **Fetched HTTP 200:** [命题式基础赛道报告模板.md](https://gitee.com/Vickyiii/hlstrack2025/raw/master/%E5%91%BD%E9%A2%98%E5%BC%8F%E5%9F%BA%E7%A1%80%E8%B5%9B%E9%81%93%E6%8A%A5%E5%91%8A%E6%A8%A1%E6%9D%BF.md).
- The README identifies itself as `嵌赛FPGA赛道-AMD 命题式基础赛道 - 初赛手册`, links that template, specifies Vitis HLS 2024.2, and describes SHA-256/LZ4/Cholesky optimization. It requests a public repository, PDF design report, code archive, and simulation/synthesis evidence.
- Provenance: this is the historical task repository identified by participant documentation, rather than a current AMD corporate rules publication. Its role and title are verified from fetched content; current organizer authority was not independently authenticated.
- Template heading: `FPGA创新设计大赛 AMD赛道命题式赛道 - 设计报告`.
- Template content SHA-256: `be45ceead0fccfd4950e1ad6a7b1afb4156e4182108758a9f072fb80fc7b9abe` (9,048 UTF-8 bytes). Gitee redirects to a signed content URL; cite the stable URL above, not the expiring redirect.

Exact second-level section names in that blank template:

1. `1. 项目概述`
2. `2. 设计原理和功能框图`
3. `3. 优化方向选择与原理`
4. `4. LLM 辅助优化记录`
5. `5. 优化前后性能与资源对比报告`
6. `6. 创新点总结`
7. `7. 遇到的问题与解决方案`
8. `8. 结论与展望`
9. `9. 参考文献`
10. `10. 附录`

Useful exact subsections: `1.3 技术规格`, `2.2 系统架构设计`, `2.3 接口设计`,
`3.3 HLS指令优化`, `5.1 测试环境`, `5.2 综合结果对比`, `5.3 详细分析`,
`5.4 正确性验证`, `5.4.1 C代码仿真结果`, `5.4.2 联合仿真结果`,
`5.4.3 硬件验证（如适用）`, `10.2 详细仿真报告`, `10.3 关键LLM交互记录`.
The comparison tables explicitly cover BRAM/DSP/LUT/FF, II, latency, throughput,
clock frequency, and resource efficiency. These are historical structural
references; neither that HLS task nor its scoring formula is established as a
requirement for the current autonomous-topic track.

### 2.2 Actual GitHub participant report: 2025, 54 pages

- **Fetched HTTP 200:** [participant README](https://raw.githubusercontent.com/hongpengWu/hlstrack2025_40562/264bb94cab8b7e10d7874b9b91d28b50f730d8c1/README.md).
- **Fetched HTTP 200 and parsed:** [hlstrack2025_40562.pdf](https://raw.githubusercontent.com/hongpengWu/hlstrack2025_40562/264bb94cab8b7e10d7874b9b91d28b50f730d8c1/hlstrack2025_40562.pdf).
- **Fetched HTTP 200:** [populated Markdown report](https://raw.githubusercontent.com/hongpengWu/hlstrack2025_40562/264bb94cab8b7e10d7874b9b91d28b50f730d8c1/%E5%91%BD%E9%A2%98%E5%BC%8F%E5%9F%BA%E7%A1%80%E8%B5%9B%E9%81%93%E6%8A%A5%E5%91%8A%E6%A8%A1%E6%9D%BF.md).
- Revision: `264bb94cab8b7e10d7874b9b91d28b50f730d8c1`. PDF: 3,238,866 bytes; SHA-256 `7990a6f36ea79bb80eddd95e9e2a5846b48e81e09bc50908d83e29c04f325ddc`.

The PDF's first two pages contain its actual contents list. It has the same
first nine section names listed in 2.1, ending at `9. 参考文献`; the blank
upstream template's section 10 is not in that contents list. Its chapter 2
subsections are `2.1 SHA-256`, `2.2 LZ4 压缩算法原理`, and
`2.3 Cholesky 分解算法原理`. Chapter 5 includes test environment, synthesis
comparisons, detailed analysis, and correctness verification.

The author's README claims a first prize. That is an author claim, not an award
verified against an organizer list; no award status is adopted here. The report
is direct evidence of what this participant wrote, not an endorsement of every
technical assertion or result in it.

### 2.3 Closely related actual report: PYNQ-Z2 audio analyzer

- **Fetched HTTP 200:** [ReadMe.md](https://raw.githubusercontent.com/VincentttWang/FPGA_SocChina_2025/67cded7e681bc2e6ad5d15c962dd8499f004764d/ReadMe.md).
- **Fetched HTTP 200 and parsed:** [doc/决赛作品简介.pdf](https://raw.githubusercontent.com/VincentttWang/FPGA_SocChina_2025/67cded7e681bc2e6ad5d15c962dd8499f004764d/doc/%E5%86%B3%E8%B5%9B%E4%BD%9C%E5%93%81%E7%AE%80%E4%BB%8B.pdf).
- Revision: `67cded7e681bc2e6ad5d15c962dd8499f004764d`. PDF: five pages, 14,902,717 bytes; SHA-256 `d04906c5beb426495e75d9f68be12ecaa3f30c9388ca85a7c6a496bb34229504`.
- Actual PDF title: `基于 PYNQ-Z2 的硬件加速实时音频分析系统`.

| Exact report section | PDF pages and actual subsections |
| --- | --- |
| `一、项目概述` | p.1: `1.1 作品简介`, `1.2 核心功能与技术栈`, `1.3 系统亮点` |
| `二、系统架构设计` | pp.1-2: `2.1 系统总体框图`, `2.2 Block Design 连线图`, `2.3 数据流说明` |
| `三、核心 IP 核设计` | pp.3-4: `3.1 FFT IP 核配置`, `3.2 FIR 滤波器 IP 核配置`, `3.3 AXI DMA 配置` |
| `四、资源使用率与功耗` | p.5 |

The README lists Vivado/Vitis HLS 2024.2 and PYNQ image v3.0.1. This is a useful
audio-system architecture/report reference, not proof that the local project
reproduces its performance or that its old tool versions satisfy today's slide.

**Actual experiment file, fetched HTTP 200:**
[notebook/FFT Experiment/fft.ipynb](https://raw.githubusercontent.com/VincentttWang/FPGA_SocChina_2025/67cded7e681bc2e6ad5d15c962dd8499f004764d/notebook/FFT%20Experiment/fft.ipynb).
JSON parsing found 13 cells and no Markdown section headings. Its code loads an
overlay, allocates buffers, transfers through AXI DMA, compares hardware output
with NumPy FFT, uses `%%timeit`, and plots the spectra. These are verified code
operations, not benchmark results reproduced in this session. The README's
actual performance subsection names are `FFT Experiment Results`,
`Resource Utilization`, and `Power Consumption`.

### 2.4 Earlier Xilinx-era participant report: 2020 provenance qualified

- **Fetched HTTP 200 and parsed:** [lch20502/PYNQ/FPGA_report.pdf](https://raw.githubusercontent.com/lch20502/PYNQ/aac0bf748d06a038d720c5313d5d6ddeb3f15e70/FPGA_report.pdf).
- **Fetched HTTP 200:** [repository page](https://github.com/lch20502/PYNQ), whose author describes it as a 2020 national FPGA competition entry.
- PDF title: `基于PYNQ和物联网的水果缺陷检测与分级装置`. Its conclusion explicitly describes using Xilinx PYNQ. The exact vendor track and any award were not independently verified.
- Revision: `aac0bf748d06a038d720c5313d5d6ddeb3f15e70`. PDF: 23 pages, 6,361,743 bytes; SHA-256 `47697327ccf5b741b37e87eeec9d2c46f8e7c883241eed768feb54051f5f5fb1`.

Exact main sections: `第一部分 设计概述` (p.1),
`第二部分 系统组成及功能说明` (p.2), `第三部分 完成情况及性能参数` (p.8),
`第四部分 总结` (p.13), `第五部分 参考文献` and `第六部分 附录` (p.14).
Section 1 includes design purpose, application field, technical characteristics,
key performance indicators, and innovation; section 4 includes
`4.1 可扩展之处` and `4.2 心得体会`. Spacing above is normalized from PDF text.
This is an additional historical structure example, not a current form.

### 2.5 Historical discovery limits

[XUP/OpenHW contest page](https://raw.githubusercontent.com/xupsh/OpenHWpages/main/contest.html)
and [gallery](https://raw.githubusercontent.com/xupsh/OpenHWpages/main/gallery.html)
both fetched HTTP 200. The contest page links `http://www.fpgachina.cn/`;
that address fetched HTTP 200 but exposed only a minimal `FPGA创新设计赛道`
shell to this client. The gallery distinguishes `OpenHW` from `FPGA创新竞赛`.
An OpenHW poster or summer-school README should not be relabeled an AMD FPGA
competition experiment report. GitHub API repository search aided discovery;
tree API requests hit an unauthenticated rate limit, so actual GitHub HTML and
raw files were used. No general web search tool was used.

## 3. Official PYNQ-Z2 and PS/PL Evidence

| Source, all fetched HTTP 200 | Verified fact | Project implication |
| --- | --- | --- |
| [AMD University Program PYNQ-Z2](https://www.amd.com/en/corporate/university-program/aup-boards/pynq-z2.html) | XC7Z020-1CLG400C Zynq-7000; dual Cortex-A9; 512 MB DDR3; 630 KB BRAM; 220 DSP slices; I2S audio/line-in; 125 MHz PL and 50 MHz PS clocks | This is a resource-constrained Zynq-7000, not an UltraScale+ or Versal board. AMD's 85K logic-cell figure is not 85K LUTs. |
| [Manufacturer PYNQ-Z2 page, linked from AMD](https://www.sparkle-embedded.com.tw/en/PYNQ-Z2) | Same device/memory/audio interfaces; 650 MHz dual-core Cortex-A9 description | CPU inference performance must be measured on this PS, separately from PC or PL performance. |
| [Official Overlay Design](https://pynq.readthedocs.io/en/latest/overlay_design_methodology/overlay_design.html) | Overlay comprises PL bitstream and HWH; matching `.bit`/`.hwh` basenames required; HWH describes IP, clocks, interrupts, resets; boot image fixes most PS configuration | Archive matching bitstream/HWH, build source, and software version; a Python call alone does not establish a deployed accelerator. |
| [Official PS/PL Interfaces](https://pynq.readthedocs.io/en/latest/overlay_design_methodology/pspl_interface.html) | MMIO gives single 32-bit transactions suited to small transfers; `allocate` provides contiguous shared buffers; DMA connects AXI streams and PS DRAM | Use registers for control/status; choose a buffered bulk path for captures. Record transport costs when reporting acceleration. |
| [Official PYNQ v3.0.1 audio driver](https://raw.githubusercontent.com/Xilinx/PYNQ/v3.0.1/pynq/lib/audio.py) | `AudioADAU1761` supports mono/stereo I2S; 24-bit samples padded to 32 bits; codec/PLL configuration uses an IIC index; default configuration rate 48 kHz | Distinguish audio transport from codec-control transactions and distinguish a default from a measured local rate. |
| [Official v3.0.1 PYNQ-Z2 base Tcl](https://raw.githubusercontent.com/Xilinx/PYNQ/v3.0.1/boards/Pynq-Z2/base/base.tcl) | Targets `xc7z020clg400-1`; includes audio controller, processing system, AXI IIC and DMA; has an explicit Vivado 2022.1 version guard | Historical example scripts require a deliberate tool/IP migration. A fetched reference is not evidence of 2026.1 compatibility. |
| [Official v3.0.1 audio pin constraints](https://raw.githubusercontent.com/Xilinx/PYNQ/v3.0.1/boards/Pynq-Z2/base/vivado/constraints/base.xdc) | Audio SCL U9, SDA T9, MCLK U5, BCLK R18, LRCLK T17, codec-to-FPGA data F17, FPGA-to-codec data G18; LVCMOS33; IIC pull-ups | Pin mapping is a reference. Pin assignments alone do not establish external setup/hold closure or correct CDC. |

The legacy TUL product URL linked from AMD failed to fetch in this session;
the fetched AMD/manufacturer pages above provide the board evidence instead.
The `master` PYNQ-Z2 base path returned 404; the verified `v3.0.1` paths above
remain readable. Historical HLS tasks using `clg484` should not be copied as
this board's physical device/package: the official board target is `clg400`.

## 4. YOLO, FINN, and Vitis AI Boundaries

### 4.1 What counts as PS inference versus PL acceleration

[Official OpenCV YOLO detector sample](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/yolo_detector.cpp)
fetched HTTP 200. It selects a backend/target and describes CPU as its default
target. PYNQ's Python environment and an ONNX file do not themselves route
inference through programmable logic.

Local inspection of `pynq/yolo_detector.py` found explicit
`DNN_BACKEND_OPENCV` and `DNN_TARGET_CPU` selection, plus the provenance string
`opencv_dnn_onnx_cpu`. This identifies the inspected path as CPU inference.
It does not establish that weights are present or that inference works on the
powered-off board. Its metadata contract accepts YOLOv8/YOLO11 detection
models; generic object detections are not verified fingertip/fret contacts.

A concrete historical example of separate work allocation is the fetched
[SR-UKF project README](https://raw.githubusercontent.com/hongpengWu/hlstrack2025_40562/264bb94cab8b7e10d7874b9b91d28b50f730d8c1/yolo_ukf_track/readme.md)
and [hardware Notebook](https://raw.githubusercontent.com/hongpengWu/hlstrack2025_40562/264bb94cab8b7e10d7874b9b91d28b50f730d8c1/yolo_ukf_track/notebooks/yolo_ukf_hardware.ipynb),
both HTTP 200. They describe YOLO plus a PL UKF accelerator; the Notebook loads
an overlay and times `ukf_step_hw`. The name `yolo_ukf_hardware` is not evidence
that YOLO convolution itself was implemented in PL.

Feasibility inference, not a benchmark: one 640 x 640 RGB uint8 image occupies
1,228,800 bytes, already larger than the board's reported 630 KB BRAM. A PL
detector therefore needs an explicit streaming/tiling/DDR plan, quantization,
operator support, and a resource budget shared with acquisition/audio logic.
No FPS, accuracy, model fit, or end-to-end speedup was measured here.

### 4.2 FINN: current support differs from older examples

- **Fetched HTTP 200, pinned official source:** [Getting Started](https://raw.githubusercontent.com/Xilinx/finn/38c42bf95abbdad493db0b0f5053060074698908/docs/finn/getting_started.rst); [rendered documentation](https://finn.readthedocs.io/en/latest/getting_started.html) also fetched HTTP 200.
- Exact section `Retired boards (Pynq-Z1/Pynq-Z2)` says both Zynq-7000 boards were retired from official support with the move to Vivado 2024.2, removed from CI/testing, and carry no guarantees. Board mappings remain and manual re-enablement is described. This is retirement of FINN board support, not retirement of xc7z020 support by Vivado.
- The documented system requirements use a Linux host, bash, Docker, and Vitis/Vivado 2024.2. The same documentation describes a local installation alternative with limitations. Docker is the primary supported path; the README's stronger Docker-only wording and the newer local-install section are not fully synchronized.
- The flow is a custom quantized network trained with Brevitas, exported to QONNX/FINN-ONNX, then transformed and built. Documentation explicitly says the aim is not to accelerate arbitrary off-the-shelf networks. A standard YOLO export is not automatically a FINN-compatible accelerator.
- **Fetched HTTP 200:** [finn-examples README](https://raw.githubusercontent.com/Xilinx/finn-examples/main/README.md) shows FINN v0.10.1, PYNQ v3.0.1 and Vivado/Vitis 2022.2 badges. Its prebuilt board list includes Pynq-Z1, Ultra96, ZCU104, U250; it explicitly says other targets such as Pynq-Z2 require build-script changes and are untested there.
- **Fetched HTTP 200:** [FINN license](https://raw.githubusercontent.com/Xilinx/finn/38c42bf95abbdad493db0b0f5053060074698908/LICENSE.txt) is BSD-3-Clause-style redistribution with retained notices, disclaimer, and non-endorsement. It is not an MIT/Apache license declaration for all generated or third-party content.

**Decision supported by these facts:** FINN on this board with the slide's
2026.1/2025.2 tools is a porting/validation task. No fetched source verifies
that combination. Keep it an optional future path until an exact model,
toolchain, resource report, and physical-board run are demonstrated.

### 4.3 Vitis AI: versioned targets and tool compatibility

- **Fetched HTTP 200:** [Vitis AI 3.5 DPU integration](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-system-integration.html). The `Zynq UltraScale+ MPSoC: DPUCZDX8G` section explicitly targets Zynq UltraScale+; other listed DPU options target Versal/Alveo. PYNQ-Z2 is Zynq-7000.
- **Fetched HTTP 200:** [Vitis AI 3.5 release notes](https://xilinx.github.io/Vitis-AI/3.5/html/docs/reference/release_notes.html). Version 3.5/DPU IP is verified with Vitis, Vivado, and PetaLinux 2023.1. That statement does not establish compatibility with 2025.2 or 2026.1.
- **Fetched HTTP 200:** [official DPU-PYNQ README](https://raw.githubusercontent.com/Xilinx/DPU-PYNQ/master/README.md). This release supports PYNQ 3.0/Vitis AI 2.5.0 and lists UltraScale+ boards including Pynq-ZU, Ultra96v2 and ZCU104; it does not list PYNQ-Z2. Pynq-ZU and Pynq-Z2 are different products.

There is no verified, ready-to-use PYNQ-Z2 DPU overlay in these official
sources. This is not a proof that all custom Zynq-7000 neural accelerators are
impossible. It is sufficient to reject a claim that installing Vitis AI or
DPU-PYNQ automatically supplies a supported Z2 YOLO accelerator.

## 5. Protocol Specifications and Evidence to Show

| Official source, fetched HTTP 200 and PDF text parsed | Verified sections/facts | Suitable experiment evidence |
| --- | --- | --- |
| [NXP UM11732, I2S bus specification](https://www.nxp.com/docs/en/user-manual/UM11732.pdf), Rev.3.0, 2022-02-17, 14 pages | pp.4-5, `3.1 Serial data`, `3.2 Word select`: two's-complement MSB first; next MSB one clock after WS change; WS=0 left/WS=1 right; leading-edge reception. `4 Timing` gives timing relationships. | Capture BCLK/LRCLK/SD with timestamps, decode channels/word width, compare known signed PCM values, inspect the one-bit delay and frame/slot boundaries. |
| [NXP UM10204, I2C-bus specification and user manual](https://www.nxp.com/docs/en/user-guide/UM10204.pdf), Rev.7.0, 2021-10-01, 62 pages | p.8 `3.1.1 SDA and SCL signals`, `3.1.3 Data validity`; p.9 `3.1.4 START and STOP conditions`, `3.1.5 Byte format`; p.10 `3.1.6 Acknowledge (ACK) and Not Acknowledge (NACK)`; section 6 electrical/timing requirements. | Capture actual SCL/SDA, decode START/address/RW/bytes/ninth-clock ACK/NACK/STOP, timestamp events, and demonstrate an injected missing-ACK condition. |

I2S sample width and slot width are distinct; do not call all 32 serial clocks
24 valid data bits. For ordinary bidirectional I2C the standard describes
pull-ups and open-drain/collector behavior; it explicitly permits a push-pull
SCL controller in the limited single-controller/no-stretching case. A report
should state supported transaction modes and clock-stretching limitations.

Local read-only findings: `rtl/i2c_write_master.v` describes itself as a small
write-only startup controller and has explicit START, bit, ACK, STOP and
bus-free states. This supports discussing FSM design; it does not establish
passive capture, arbitrary read transactions, or a general analyzer. Existing
I2S simulation files and audio XDCs provide places to collect evidence, but
were not executed in this research session. The inspected audio XDCs contain
BCLK clocks/asynchronous groups and no `set_input_delay`/`set_output_delay`.
Positive internal timing slack therefore must not be reported as a completed
external I2S timing/CDC sign-off.

## 6. Chord, Scale, and Fingering References With Licenses

### 6.1 Preferred permissive upstream references

| Upstream evidence, all fetched HTTP 200 | License verified from actual file | Reusable facts and limits |
| --- | --- | --- |
| [Tonal chord-type data](https://raw.githubusercontent.com/tonaljs/tonal/e555422aef910daa87cb924028f71f0a013f3ed7/packages/chord-type/data.ts), [chord-type LICENSE](https://raw.githubusercontent.com/tonaljs/tonal/e555422aef910daa87cb924028f71f0a013f3ed7/packages/chord-type/LICENSE) | MIT; copyright 2015 danigb | Intervals/name/aliases dictionary, e.g. major `1P 3M 5P`, minor `1P 3m 5P`. It is music theory, not audio transcription. The file records a Wikibooks source; preserve provenance when reusing data. |
| [Tonal scale-type data](https://raw.githubusercontent.com/tonaljs/tonal/e555422aef910daa87cb924028f71f0a013f3ed7/packages/scale-type/data.ts), [scale-type LICENSE](https://raw.githubusercontent.com/tonaljs/tonal/e555422aef910daa87cb924028f71f0a013f3ed7/packages/scale-type/LICENSE) | MIT; copyright 2015 danigb | Major/Ionian, minor/Aeolian, pentatonic, Dorian and other interval sets. Shared pitch classes do not uniquely establish tonic or mode. |
| [chords-db guitar JSON](https://raw.githubusercontent.com/tombatossals/chords-db/df06fa7b425cf5fd29485ff6591236b3557e3fac/lib/guitar.json), [LICENSE](https://raw.githubusercontent.com/tombatossals/chords-db/df06fa7b425cf5fd29485ff6591236b3557e3fac/LICENSE) | MIT; copyright 2016 David Rubert | Explicit standard tuning `E2 A2 D3 G3 B3 E4`; positions contain frets, fingers, baseFret, barres, MIDI. First Dsus2 position is `[-1,-1,0,2,3,0]`, fingers `[0,0,0,2,3,0]`, baseFret 1. This is a reference fingering catalogue, not evidence of the player's actual contacts. |

Tonal revision: `e555422aef910daa87cb924028f71f0a013f3ed7`.
chords-db revision: `df06fa7b425cf5fd29485ff6591236b3557e3fac`.
The fetched guitar JSON SHA-256 is
`cfe439962b2f444d2c341b1f0261403b4c3a3416e321147286fc608922699974`.

Use the pinned JSON/source schema instead of copying its README's old Dsus2
example: the fetched README shows `0320xx`, while the actual first source
position is `xx0230`. Respect low-to-high tuning order, muted/open strings,
relative positive frets plus `baseFret`, and distinguish finger numbers from
fret numbers. Validate individual voicings before presenting them as a result.
Keep MIT copyright/permission notices with copied substantial data/code.

### 6.2 Existing conceptual references need distinct attribution

| Actual fetched license file, HTTP 200 | License evidence | Implication for an MIT/Apache-2.0 entry |
| --- | --- | --- |
| [librosa LICENSE.md](https://raw.githubusercontent.com/librosa/librosa/main/LICENSE.md) | ISC | Permissive reference for pitch/chroma/time-series methods, with its own notice retained. It is not itself MIT/Apache. |
| [Essentia COPYING.txt](https://raw.githubusercontent.com/MTG/essentia/master/COPYING.txt) | GNU AGPL v3 text | Do not silently import or relicense its implementation as MIT/Apache. Conceptual citation is distinct from code reuse; exact file headers govern version-or-later terms. |
| [NNLS-Chroma/Chordino COPYING](https://raw.githubusercontent.com/c4dm/nnls-chroma/master/COPYING) | GNU GPL v2 text | Same separation of conceptual reference and copied implementation; inspect exact file headers before any reuse. |
| [Ultralytics LICENSE](https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE), [YOLOv5 LICENSE](https://raw.githubusercontent.com/ultralytics/yolov5/master/LICENSE) | Both fetched versions carry GNU AGPL v3 text | The word YOLO does not establish a permissive license. Check the exact implementation, weights and dataset terms independently before including them in a submission. |

The screenshot's MIT/Apache-2.0 requirement concerns the submitted work; it does
not prove that every third-party component is forbidden or automatically
compatible. Keep a component/license inventory and original notices rather
than claiming all references share the project's license. No model, weights,
dataset, or library was imported into this project by this research.

## 7. Proposed Local Report Shape and Acceptance Evidence

This outline is a recommendation derived from the fetched examples and the
screenshots. It is not an official current competition template.

1. **项目概述与报名题目对应**: retain `FPGA信号采集及协议分析仪`; define acquisition channels, interfaces, trigger/capture depth and supported decoders; identify music/vision as the application.
2. **系统架构与软硬件分工**: block diagram from codec/pins through capture/decoder/buffer/AXI to PS/UI; identify PL FFT and CPU vision separately; list bitstream/HWH and tool versions.
3. **逻辑设计、状态机与接口协议**: I2C FSM/ACK errors; I2S serial timing/channel reconstruction; capture/trigger/buffer control and overrun handling; registers and transport format.
4. **时序约束与跨时钟域**: clock sources, timing budgets, I/O delays, reset crossings, CDC mechanism and reports; list unresolved constraints explicitly.
5. **实验方法与结果**: known I2S frames and I2C transactions, negative cases, independent expected decoder outputs; synthesis/implementation evidence; separate kernel, transport and whole-system latency; separate recorded historical evidence from fresh tests.
6. **应用演示与识别边界**: waveform and transaction display first, then FFT/chord/scale and camera findings; distinguish synthetic stimuli, reference fingerings, model proposals, and observed contacts; no unsupported accuracy claims.
7. **资源、版本与复现**: LUT/FF/BRAM/DSP, timing, build commands, exact package/device, software versions, file hashes and startup procedure; migrate historical examples to 2026.1/2025.2 deliberately.
8. **创新点、问题、结论与参考文献**: compare to stated targets and a defined software baseline; provide an honest completed/pending table, source/license inventory, and raw evidence appendix.

Board-off stop condition: report only inspected source facts and existing
artifacts as historical evidence. A later board session is required to verify
pin captures, actual ACK/NACK behavior, end-to-end acquisition, deployed model
readiness, and any performance/accuracy claims. Simulation alone cannot meet
the slide's on-board demonstration requirement.

## 8. Verification Limits

- Final HTTP citation check completed at 2026-10-15 16:43:16 UTC, about 12 minutes after research began: all 42 distinct linked HTTP sources returned 200. All three report PDFs had valid PDF headers and the recorded SHA-256 values. Both local screenshot links resolve; the UTF-8 report contains no replacement characters.
- No current deadline, advancement, acceptance, prize, or scoring weights were established. No participant award claim was promoted to organizer-confirmed status.
- The historical template's mandatory HLS/LLM structure is not assumed mandatory for the current autonomous-topic initial group.
- Report PDFs above were fetched and their text/section structure parsed; they were not executed, reproduced, or audited for all technical claims.
- Official hardware/docs and upstream licenses support the stated constraints. Mutable `latest`/`main` documents can change; use pinned sources where supplied and recheck exact tool/model combinations before implementation claims.
- Research changed only `reports/RESEARCH_SOURCES.md`; it did not power or access the board, run Vivado, alter RTL/software, or establish fresh board results.
