from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
PROJECT = Path(os.environ.get("FRETLENS_PROJECT_ROOT", ROOT.parent.parent)).resolve()
OUTPUT = ROOT / "基于PYNQ-Z2的吉他指法与和弦识别视觉子系统实验报告.docx"
REPOSITORY_URL = os.environ.get(
    "REPORT_REPOSITORY_URL",
    "https://github.com/SunRuoxuan18/guitar-vision-fpga",
).rstrip("/")

FONT_REGULAR = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

BLACK = RGBColor(0, 0, 0)
SLATE = "2F4F66"
PALE_BLUE = "EAF0F5"
LIGHT_GRAY = "F4F5F6"
MID_GRAY = "6B737A"
BORDER_GRAY = "D9D9D9"


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    size: int,
    bold: bool = False,
    fill: str = "black",
) -> None:
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    font = pil_font(size, bold)
    spacing = int(size * 0.45)
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        heights.append(bbox[3] - bbox[1])
    total = sum(heights) + spacing * (len(lines) - 1)
    y = y0 + (y1 - y0 - total) / 2
    for line, height in zip(lines, heights):
        width = draw.textlength(line, font=font)
        draw.text((x0 + (x1 - x0 - width) / 2, y), line, fill=fill, font=font)
        y += height + spacing


def create_assets() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    # Figure 1: system and data flow.
    image = Image.new("RGB", (1800, 970), "white")
    draw = ImageDraw.Draw(image)
    stages = [
        ("C922 / USB", "640x480\nYUYV"),
        ("板卡 PS", "颜色/边界候选\n端点与几何校验"),
        ("板卡 PL", "灰度、金字塔 LK\n卷积加速"),
        ("视觉估计", "手部 ROI\n四指尖热图"),
        ("指板几何", "六弦、品丝\n单应校正"),
        ("音符投影", "弦号 + 品号\nMIDI 候选"),
        ("网页显示", "视频、线段、ROI\n指尖与候选音符"),
    ]
    box_w, box_h = 205, 150
    gap = 39
    x = 30
    y = 120
    for index, (title, detail) in enumerate(stages):
        draw.rounded_rectangle(
            (x, y, x + box_w, y + box_h),
            radius=12,
            fill="white",
            outline="#B8C3CC",
            width=3,
        )
        draw_centered(draw, (x + 8, y + 10, x + box_w - 8, y + 53), title, 27, True)
        draw_centered(draw, (x + 8, y + 54, x + box_w - 8, y + box_h - 8), detail, 22)
        if index < len(stages) - 1:
            x1 = x + box_w
            draw.line((x1 + 7, y + box_h // 2, x1 + gap - 12, y + box_h // 2), fill="#344B5C", width=5)
            draw.polygon(
                [
                    (x1 + gap - 3, y + box_h // 2),
                    (x1 + gap - 18, y + box_h // 2 - 9),
                    (x1 + gap - 18, y + box_h // 2 + 9),
                ],
                fill="#344B5C",
            )
        x += box_w + gap

    draw.text((40, 30), "视觉数据流与 PS/PL 分工", fill="black", font=pil_font(38, True))
    draw.text(
        (40, 315),
        "并行校验：帧号、流号、几何代次和结果年龄。过期结果不得显示为当前帧证据。",
        fill="#344B5C",
        font=pil_font(25),
    )
    y2 = 420
    draw.rounded_rectangle((80, y2, 815, y2 + 180), radius=13, fill="#" + PALE_BLUE, outline="#A9B7C2", width=3)
    draw.text((110, y2 + 24), "PS 路径", fill="black", font=pil_font(30, True))
    draw.text(
        (110, y2 + 74),
        "颜色区域、边界候选、RANSAC、几何约束、音符映射、调度与失效拒绝",
        fill="black",
        font=pil_font(23),
    )
    draw.rounded_rectangle((985, y2, 1720, y2 + 180), radius=13, fill="#" + PALE_BLUE, outline="#A9B7C2", width=3)
    draw.text((1015, y2 + 24), "PL 路径", fill="black", font=pil_font(30, True))
    draw.text(
        (1015, y2 + 74),
        "灰度/边缘原语、稀疏光流、现有四指尖网络卷积 MAC",
        fill="black",
        font=pil_font(23),
    )
    draw.text((80, 700), "目标条件", fill="black", font=pil_font(31, True))
    draw.text(
        (80, 755),
        "30 FPS 需要约 33 ms 的完整新帧预算。原型当前仍是“相机较低帧率 + 异步分析结果”，",
        fill="black",
        font=pil_font(24),
    )
    draw.text(
        (80, 802),
        "神经网络、跟踪、几何和网页刷新不能只按单个模块耗时相加，而要看端到端 p95 延迟。",
        fill="black",
        font=pil_font(24),
    )
    image.save(ASSETS / "vision_pipeline.png")

    # Figure 2: copy the reviewed endpoint-band frame at readable resolution.
    endpoint = PROJECT / "diagnostics/endpoint_tracking_current_20261030/WIN_20261026_01_18_35_Pro-endpoint_bands.jpg"
    with Image.open(endpoint) as im:
        im.resize((1280, 720), Image.Resampling.LANCZOS).save(ASSETS / "endpoint_tracking.png", quality=94)

    # Figure 3: measured fret/string rendering.
    fret = PROJECT / "diagnostics/endpoint_tracking_20261030/fret_rendering_verified_20261030.jpg"
    with Image.open(fret) as im:
        im.resize((960, 720), Image.Resampling.LANCZOS).save(ASSETS / "fret_string_overlay.png", quality=94)

    # Figure 4: latest reviewed model metrics and per-finger comparison.
    chart = Image.new("RGB", (1800, 790), "white")
    d = ImageDraw.Draw(chart)
    d.text((35, 20), "复训前后指尖验证结果", fill="black", font=pil_font(34, True))
    d.text((35, 68), "同一 98 帧验证集，356 个可见指尖点；PCK 为归一化 ROI 距离指标", fill="#4E5961", font=pil_font(22))

    chart_x, chart_y, chart_w, chart_h = 85, 140, 760, 500
    d.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill="#777777", width=2)
    d.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill="#777777", width=2)
    for tick in range(0, 101, 20):
        ytick = chart_y + chart_h - chart_h * tick / 100
        d.line((chart_x, ytick, chart_x + chart_w, ytick), fill="#E3E3E3", width=1)
        d.text((chart_x - 45, ytick - 12), f"{tick}%", fill="#4E5961", font=pil_font(19))
    groups = [
        ("PCK@0.05", 72.75, 73.60),
        ("PCK@0.10", 89.89, 91.29),
    ]
    bar_w = 95
    for i, (label, before, after) in enumerate(groups):
        center = chart_x + 210 + i * 330
        for j, (value, color) in enumerate(((before, "9AA8B3"), (after, SLATE))):
            x0 = center - bar_w - 12 + j * (bar_w + 24)
            y0 = chart_y + chart_h - chart_h * value / 100
            d.rounded_rectangle((x0, y0, x0 + bar_w, chart_y + chart_h), radius=6, fill=f"#{color}")
            d.text((x0 + 14, y0 - 33), f"{value:.2f}", fill="black", font=pil_font(21, True))
        d.text((center - 90, chart_y + chart_h + 18), label, fill="black", font=pil_font(22, True))
    d.rounded_rectangle((chart_x + 65, chart_y + 18, chart_x + 90, chart_y + 42), radius=3, fill="#9AA8B3")
    d.text((chart_x + 102, chart_y + 13), "原模型", fill="black", font=pil_font(20))
    d.rounded_rectangle((chart_x + 235, chart_y + 18, chart_x + 260, chart_y + 42), radius=3, fill="#" + SLATE)
    d.text((chart_x + 272, chart_y + 13), "10 月 30 日复训", fill="black", font=pil_font(20))

    px, py, pw, ph = 1010, 140, 720, 500
    d.line((px, py + ph, px + pw, py + ph), fill="#777777", width=2)
    d.line((px, py, px, py + ph), fill="#777777", width=2)
    for tick in range(0, 101, 20):
        ytick = py + ph - ph * tick / 100
        d.line((px, ytick, px + pw, ytick), fill="#E3E3E3", width=1)
        d.text((px - 43, ytick - 12), f"{tick}%", fill="#4E5961", font=pil_font(19))
    fingers = [
        ("食指", 80.61, 80.61),
        ("中指", 58.16, 64.29),
        ("无名指", 74.49, 69.39),
        ("小指", 80.65, 83.87),
    ]
    bar_w2 = 63
    for i, (label, before, after) in enumerate(fingers):
        center = px + 105 + i * 170
        for j, (value, color) in enumerate(((before, "9AA8B3"), (after, SLATE))):
            x0 = center - bar_w2 - 8 + j * (bar_w2 + 16)
            y0 = py + ph - ph * value / 100
            d.rounded_rectangle((x0, y0, x0 + bar_w2, py + ph), radius=5, fill=f"#{color}")
        d.text((center - 52, py + ph + 18), label, fill="black", font=pil_font(20, True))
    d.text((px + 8, py - 58), "分指 PCK@0.05", fill="black", font=pil_font(25, True))
    d.text((px + 8, py + ph + 62), "中指改善最大，但仍是四指中最弱项；无名指出现回退。", fill="#4E5961", font=pil_font(21))
    d.text((35, 735), "结论：增加样本带来小幅提升，但没有消除自动 ROI、遮挡和指间混淆造成的结构性误差。", fill="black", font=pil_font(23))
    chart.save(ASSETS / "fingertip_metrics.png")

    # Figure 5: two representative rows from the same-ROI comparison.
    source = PROJECT / "experiments/handpose_x_trial/output/three_way_comparison_sheet.jpg"
    with Image.open(source) as im:
        row_h = im.height // 8
        rows = [2, 7]
        crops = [im.crop((0, index * row_h, im.width, (index + 1) * row_h)) for index in rows]
        combined = Image.new("RGB", (im.width, row_h * 2 + 24), "white")
        combined.paste(crops[0], (0, 0))
        combined.paste(crops[1], (0, row_h + 24))
        combined.resize((1260, 842), Image.Resampling.LANCZOS).save(
            ASSETS / "handpose_comparison.png", quality=93
        )


def set_run_font(run, size: float | None = None, bold: bool | None = None, east: str = "宋体") -> None:
    run.font.name = "Times New Roman"
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    run.font.color.rgb = BLACK
    r_pr = run._element.get_or_add_rPr()
    fonts = r_pr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, fonts)
    fonts.set(qn("w:eastAsia"), east)
    fonts.set(qn("w:ascii"), "Times New Roman")
    fonts.set(qn("w:hAnsi"), "Times New Roman")
    for key in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        fonts.attrib.pop(qn(key), None)


def set_style(style, size: float, bold: bool = False, east: str = "宋体") -> None:
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = BLACK
    r_pr = style.element.get_or_add_rPr()
    fonts = r_pr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, fonts)
    fonts.set(qn("w:eastAsia"), east)
    fonts.set(qn("w:ascii"), "Times New Roman")
    fonts.set(qn("w:hAnsi"), "Times New Roman")
    for key in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        fonts.attrib.pop(qn(key), None)
    for border in list(style.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)


def add_page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    set_run_font(run, 9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    run._r.addnext(field)


def add_body(document: Document, text: str, first_indent: bool = True, after: float = 5) -> None:
    p = document.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.275 if first_indent else 0)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.keep_together = True
    run = p.add_run(text)
    set_run_font(run, 10.8)


def add_lead_paragraph(document: Document, lead: str, text: str) -> None:
    p = document.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.275)
    p.paragraph_format.space_after = Pt(5)
    leading = p.add_run(lead)
    set_run_font(leading, 10.8, True)
    body = p.add_run(text)
    set_run_font(body, 10.8)


def add_h1(document: Document, text: str, page_break: bool = False) -> None:
    p = document.add_paragraph(style="Heading 1")
    if page_break:
        p.paragraph_format.page_break_before = True
    p.add_run(text)


def add_h2(document: Document, text: str) -> None:
    p = document.add_paragraph(style="Heading 2")
    p.add_run(text)


def add_heading_caption(document: Document, text: str) -> None:
    p = document.add_paragraph(style="Caption")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_run_font(run, 9.2, True)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shade = OxmlElement("w:shd")
    shade.set(qn("w:fill"), fill)
    tc_pr.append(shade)


def no_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def add_table(
    document: Document,
    caption: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    font_size: float = 9.2,
) -> None:
    add_heading_caption(document, caption)
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), BORDER_GRAY)
        borders.append(element)
    table._tbl.tblPr.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for edge, width in (("top", "70"), ("bottom", "70"), ("left", "85"), ("right", "85")):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:w"), width)
        element.set(qn("w:type"), "dxa")
        margins.append(element)
    table._tbl.tblPr.append(margins)

    header_row = table.rows[0]
    no_split(header_row)
    header_row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for cell, header, width in zip(header_row.cells, headers, widths):
        cell.width = Inches(width)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, SLATE)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.keep_together = True
        run = p.add_run(header)
        set_run_font(run, font_size, True)
        run.font.color.rgb = RGBColor(255, 255, 255)

    for row_index, values in enumerate(rows):
        row = table.add_row()
        no_split(row)
        fill = "FFFFFF" if row_index % 2 == 0 else LIGHT_GRAY
        for cell, value, width in zip(row.cells, values, widths):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_shading(cell, fill)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            p.paragraph_format.keep_together = True
            if row_index < len(rows) - 1:
                p.paragraph_format.keep_with_next = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(value) <= 15 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(value)
            set_run_font(run, font_size)
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(0)


def add_figure(document: Document, filename: str, width: float, caption: str, alt: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    shape = p.add_run().add_picture(str(ASSETS / filename), width=Inches(width))
    shape._inline.docPr.set("descr", alt)
    cp = document.add_paragraph(style="Caption")
    cp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cp.paragraph_format.space_after = Pt(7)
    run = cp.add_run(caption)
    set_run_font(run, 9.1)


def add_hyperlink(
    paragraph,
    text: str,
    url: str,
    font_size: float = 9.2,
) -> None:
    relationship_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)

    run = OxmlElement("w:r")
    run_properties = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), "Microsoft YaHei")
    fonts.set(qn("w:hAnsi"), "Microsoft YaHei")
    fonts.set(qn("w:eastAsia"), "微软雅黑")
    run_properties.append(fonts)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), str(round(font_size * 2)))
    run_properties.append(size)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1F4E79")
    run_properties.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    run_properties.append(underline)
    run.append(run_properties)

    text_element = OxmlElement("w:t")
    text_element.text = text
    run.append(text_element)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def configure_document() -> Document:
    document = Document()
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.88)
    section.right_margin = Inches(0.88)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)
    section.different_first_page_header_footer = True

    set_style(document.styles["Normal"], 10.8)
    normal = document.styles["Normal"].paragraph_format
    normal.line_spacing = 1.25
    normal.space_after = Pt(5)
    normal.widow_control = True

    set_style(document.styles["Title"], 25, True, "微软雅黑")
    title = document.styles["Title"].paragraph_format
    title.space_before = Pt(0)
    title.space_after = Pt(10)
    title.keep_with_next = True
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    set_style(document.styles["Heading 1"], 15.5, True, "微软雅黑")
    h1 = document.styles["Heading 1"].paragraph_format
    h1.space_before = Pt(5)
    h1.space_after = Pt(8)
    h1.keep_with_next = True

    set_style(document.styles["Heading 2"], 11.8, True, "微软雅黑")
    h2 = document.styles["Heading 2"].paragraph_format
    h2.space_before = Pt(7)
    h2.space_after = Pt(4)
    h2.keep_with_next = True

    set_style(document.styles["Caption"], 9.1)
    document.styles["Caption"].paragraph_format.line_spacing = 1.12

    add_page_field(section.footer.paragraphs[0])
    section.first_page_footer.is_linked_to_previous = False
    document.core_properties.title = "基于 PYNQ-Z2 的吉他指法与和弦识别视觉子系统实验报告"
    document.core_properties.subject = "计算机视觉、嵌入式 FPGA、音乐信息检索"
    document.core_properties.author = "项目负责人"
    document.core_properties.keywords = "PYNQ-Z2; 指板检测; 手部关键点; 指尖; FPGA; 和弦识别"
    return document


def build_cover(document: Document) -> None:
    p = document.add_paragraph()
    p.paragraph_format.space_before = Pt(72)
    p.paragraph_format.space_after = Pt(18)
    run = p.add_run("实验研究报告")
    set_run_font(run, 12.5, True, "微软雅黑")
    run.font.color.rgb = RGBColor(64, 78, 89)

    title = document.add_paragraph(style="Title")
    title.add_run("基于 PYNQ-Z2 的吉他指法与\n和弦识别视觉子系统")

    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(40)
    subtitle.paragraph_format.keep_with_next = True
    run = subtitle.add_run("从琴颈定位、四指尖估计到音符投影的原型实现、实测结果与后续研究计划")
    set_run_font(run, 14, False, "微软雅黑")
    run.font.color.rgb = RGBColor(45, 55, 63)

    add_body(
        document,
        "本报告说明项目在比赛截止前实际完成的视觉工作、可复核测量、未能闭环的原因，以及可以继续验证的工程问题。"
        "报告不把离线回放、合成测试或模块级通过写成真实演奏准确率。",
        first_indent=False,
        after=50,
    )

    details = [
        ("项目阶段", "比赛原型与可行性研究，未完成实时和弦级验收"),
        ("应用场景", "木吉他按弦手与指板的单目视觉分析"),
        ("硬件平台", "PYNQ-Z2 / XC7Z020、Logitech C922"),
        ("汇报日期", "2026 年 11 月 9 日"),
        ("提交人", "项目负责人"),
    ]
    for label, value in details:
        p = document.add_paragraph()
        p.paragraph_format.space_after = Pt(7)
        label_run = p.add_run(f"{label}　")
        set_run_font(label_run, 10, True, "微软雅黑")
        value_run = p.add_run(value)
        set_run_font(value_run, 10)


def build_summary(document: Document) -> None:
    add_h1(document, "摘要", page_break=True)
    add_body(
        document,
        "本项目尝试在资源受限的 PYNQ-Z2 上建立一条可解释的吉他视觉识别链路：从 C922 摄像头采集指板画面，"
        "先获得琴颈和指板几何，再定位手部与四个指尖，把指尖投影到实测的六根琴弦和品丝边界上，得到弦号、品号和"
        "MIDI 音符候选，最后与音频谐波分析结果融合并显示到网页。研究目标不是单纯训练一个黑盒分类器，而是建立"
        "“人体动作、乐器几何、声音证据”之间可检查的中间表示。",
    )
    add_body(
        document,
        "比赛阶段已经完成并保存了多个可运行原型：颜色/边界候选与端点式琴颈跟踪、PL 金字塔 Lucas-Kanade 光流、"
        "PS RANSAC 与当前帧支持校验、208×208 四指尖热图网络、实测弦品几何、网页叠加和音视频数据身份校验。"
        "独立四指尖跟踪约 30.54 ms，音符投影后期约 4.3 至 5.3 ms，音频融合约 0.410 ms；但完整四指尖神经网络"
        "一次调用中位数仍为 1104 ms，启用网页分析时实拍采集约 7.61 FPS，尚未达到实时要求。",
    )
    add_body(
        document,
        "10 月 30 日用于对比的复训在 98 帧、356 个可见指尖点上，把 PCK@0.05 从 72.75% 提升到 73.60%，"
        "PCK@0.10 从 89.89% 提升到 91.29%。提升是真实的，但幅度有限，中指仍是最弱项，而且该指标只衡量点距离，"
        "不能代表弦号、品号或和弦正确率。琴颈首次锁定在回放中仍需约 1.3 至 3.2 s，长距离运动和遮挡后的重获仍不稳定。"
    )
    add_body(
        document,
        "因此，项目没有完成“实时稳定识别任意和弦”的最终验收。主要障碍不是某一个模型参数，而是视觉与系统的耦合问题："
        "自动手部 ROI 与人工标注 ROI 不一致、热图网络缺少可靠的可见性判断、长琴颈尺度变化导致跟踪漂移、"
        "弦品几何缺少独立真值、异步结果造成时序错位，以及板卡上网络调度和 DDR 传输成本过高。这些问题构成了"
        "可以继续研究的明确课题，而不是只能通过继续堆标注图片解决。",
    )
    add_body(
        document,
        "关键词：计算机视觉；PYNQ-Z2；指板检测；手部关键点；指尖定位；视觉测量；音视频融合；和弦识别",
        first_indent=False,
        after=9,
    )
    add_h2(document, "核心结论")
    add_lead_paragraph(
        document,
        "已完成：",
        "形成了从相机、板卡处理到网页可视化的完整实验骨架，保存了可追溯的模型、代码和测量证据。",
    )
    add_lead_paragraph(
        document,
        "未完成：",
        "尚未做到 30 FPS、稳定的全指板弦品格、可靠的遮挡可见性和真实演奏下的和弦级准确率。",
    )
    add_lead_paragraph(
        document,
        "最有价值的下一步：",
        "把“继续增加图片”改为“自动 ROI 与人工 ROI 配对评测 + 可见性感知的手部关键点 + 端点式琴颈跟踪 + "
        "弦品几何真值 + 延迟预算”的系统研究。",
    )


def build_section_1(document: Document) -> None:
    add_h1(document, "1 研究背景与问题定义", page_break=True)
    add_body(
        document,
        "吉他和弦识别通常从音频入手，但仅依赖声音很难区分相似音级集合、转位、横按、开放弦和同一和弦的不同把位。"
        "视觉能够提供另一类证据：按弦手落在哪些弦和品上，当前是否发生换把，手指是否形成横按或开放指型。"
        "将视觉位置与音频音高联合，有望减少纯音频模型在弱起音、扫弦瞬态、混响和多人环境中的歧义。",
    )
    add_body(
        document,
        "本项目把问题拆成五个连续子问题。第一，找到画面中的琴颈和指板边界；第二，从指板几何中恢复六根弦与品丝；"
        "第三，定位按弦手及其四个指尖；第四，把指尖映射到弦品坐标和音符候选；第五，把视觉候选与音频音级集合"
        "进行概率化或规则化融合，输出和弦。与直接做图像到和弦标签的端到端网络相比，这种设计更容易定位失败来源，"
        "也更适合在 FPGA 上按不同更新频率分配任务。",
    )
    add_body(
        document,
        "系统约束非常明确：目标板是 PYNQ-Z2，PL 为 XC7Z020；摄像头输入为 YUYV 640×480；网页需要持续显示视频、"
        "琴颈边界、手部 ROI、指尖和弦品结果；若以 30 FPS 为实时目标，每帧全部处理预算约为 33 ms。"
        "实际系统还存在浏览器编码、USB 采集、JPEG 传输、ARM 调度和结果排队，因此单个卷积模块小于 33 ms "
        "并不能证明整机实时。",
    )
    add_body(
        document,
        "本文所说的“完成”采用系统级标准：在独立演奏视频上，琴颈和指板在运动、遮挡和重新进入时保持可用；"
        "自动 ROI 下的四指位置达到可映射到正确弦品的误差；弦品格持续有效；音视频融合后的和弦在带时间戳的数据集上"
        "有可复现的准确率和延迟分布。按这一标准，项目仍处于原型研究阶段。",
    )


def build_section_2(document: Document) -> None:
    add_h1(document, "2 系统架构与数据流")
    add_body(
        document,
        "视觉系统采用 PS 与 PL 协同结构。PS 运行 Linux、OpenCV、候选筛选、RANSAC、几何测量、结果身份校验和网页接口；"
        "PL 负责规则图像原语、稀疏光流和已部署网络的卷积乘加。四指尖神经网络并非在 PL 内独立运行完整图，"
        "ARM 仍负责层调度、张量准备和热图解码。这一区别对性能判断很重要：网络“位于 PL”只说明主要 MAC 在可编程逻辑中执行，"
        "不说明从输入到输出的全部路径都已硬件流水化。",
    )
    add_figure(
        document,
        "vision_pipeline.png",
        6.55,
        "图 1　视觉子系统的数据流、PS/PL 分工与目标时延条件。图中“校验”不是装饰层，而是防止旧帧结果覆盖当前画面的必要机制。",
        "视觉数据流和 PS/PL 分工示意",
    )
    add_body(
        document,
        "每个请求都携带流号、代次和帧序号。前端只接受与当前分析帧身份一致的结果；几何丢失、网络超时或结果年龄过大时，"
        "系统清除旧位置而不是继续显示。该机制在调试时降低了“看起来在跟踪、实际是历史结果”的风险，也解释了为什么某些帧"
        "宁可显示搜索状态，也不显示看似连续但不可信的框。",
    )
    add_body(
        document,
        "PS 与 PL 之间的传输成本是当前系统的关键瓶颈之一。现有图包含 21 个 PL 卷积层和 14 条传输边，"
        "测得这些 PL 神经事务合计约 395 至 404 ms，尚未包括其余调度、预处理和数据搬运。"
        "因此，即使 PL 中单个卷积很快，完整四指尖网络调用的中位数仍达到 1104 ms，无法仅靠调整 Python 轮询解决。",
    )


def build_section_3(document: Document) -> None:
    add_h1(document, "3 琴颈与指板几何获取")
    add_h2(document, "3.1 自动候选与端点式跟踪")
    add_body(
        document,
        "早期方案使用边缘、Hough 线段和透视约束寻找琴颈，但在手部遮挡、琴头边界和长距离运动中容易把多个纹理区域连接成错误四边形。"
        "后续改为颜色与边界候选：在缩小后的图像中寻找两条长边界、内部颜色支撑和横向品丝响应，再交给 PL 金字塔光流跟踪；"
        "PS 使用 RANSAC 和当前帧特征支持拒绝不一致的形变。",
    )
    add_body(
        document,
        "为了提高长琴颈的鲁棒性，进一步采用“只定位画面中露出的头和尾”的端点式思路：分别估计琴颈上下边界的远端与近端，"
        "再用两条长线连接。这样中间即使被手挡住，也不必要求整条边界全部可见。靠近琴头区域还加入上边界修正，"
        "避免直线越过琴枕延伸到琴头。图 2 展示了该表示方式在真实吉他画面中的实例。",
    )
    add_figure(
        document,
        "endpoint_tracking.png",
        6.35,
        "图 2　端点式琴颈表示。黄色两条长线表示指板上下边界，黄色短边仅用于闭合可见区域；该图来自真实录制帧的回放与人工检查。",
        "端点式琴颈上下边界跟踪画面",
    )
    add_body(
        document,
        "端点表示的优点是把“琴颈整体检测”转化为少量几何点的稳定性问题，并且可以直接为后续单应变换和六弦插值提供边界。"
        "缺点仍然是尺度变化和遮挡：当画面只露出很短的琴颈段，两个远端的估计误差会被长直线放大；当手快速运动时，"
        "光流特征混合手指纹理，边界会被向内或向外拉偏。单纯降低检测阈值只会让错框更频繁，不能解决数据关联问题。",
    )
    add_h2(document, "3.2 实测结果与限制")
    add_table(
        document,
        "表 1　琴颈跟踪与颜色边界获取的代表性回放结果",
        ["测量项", "改进前", "改进后", "证据边界"],
        [
            ["首次获得可跟踪琴颈", "2.281 s", "1.328 s", "真实录制帧回放，非实时首次开机"],
            ["24 帧运动序列保留琴颈 ROI", "20/24", "23/24", "可用率，不是边界 IoU"],
            ["24 帧运动序列保留手部 ROI", "20/24", "23/24", "手部 ROI 仍为轻量肤色/几何建议"],
            ["请求往返中位数", "273.5 ms", "266.0 ms", "不是端到端显示 FPS"],
            ["请求往返 p95", "见原始样本", "812.85 ms", "失败恢复请求拉高尾部延迟"],
            ["黑帧保留陈旧琴颈", "0/6", "0/6", "能拒绝无图像支持的旧几何"],
        ],
        [1.75, 1.1, 1.1, 2.55],
        8.8,
    )
    add_body(
        document,
        "48 张吉他采样帧全部产生候选，8 张无吉他帧和 2 张房间负样本均未产生候选。这一结果说明颜色/边界方法具备可用性，"
        "但样本来自少量连续视频，且没有独立标定的边界真值，因此不能报告为“琴颈检测准确率”。"
        "另外，首次锁定仍需约 1.3 s，端到端运动请求的 p95 可超过 0.8 s，算法在交互上仍明显慢于视觉直觉。",
    )


def build_section_4(document: Document) -> None:
    add_h1(document, "4 手部 ROI 与四指尖估计")
    add_h2(document, "4.1 网络结构与运行方式")
    add_body(
        document,
        "指尖模型使用 MobileNetV2 手部关键点骨干和轻量 stride-8 热图头，输入 208×208，输出四张指尖热图。"
        "模型约 5.65 万参数，选择该结构的目的是让卷积 MAC 能够在现有 PL 缓冲和资源预算中执行。"
        "训练目标只监督可见指尖点，测试时对每张热图取峰值，始终返回四个候选位置。该设计适合演示，但并不提供"
        "“这个指尖是否可见、是否真的按在弦上”的可靠概率，因此不能把四个峰值直接当作四个按弦事实。",
    )
    add_h2(document, "4.2 新增数据与复训结果")
    add_body(
        document,
        "10 月 30 日从三段 C922 录制中均匀抽取 1000 帧，原始分辨率为 1920×1080。该批素材与已有数据结合后，"
        "复训记录使用 781 张训练、98 张验证；数据集目录总计 1004 张图，但并非全部图像都在本次训练/验证清单中。"
        "只有验证集参与下面的精度报告。由于验证仍来自同一演奏者和相近拍摄环境，结果不能外推为跨人泛化。",
    )
    add_figure(
        document,
        "fingertip_metrics.png",
        6.4,
        "图 3　复训前后的指尖验证指标。左图为整体 PCK，右图为分指 PCK@0.05；指标为 ROI 归一化点距离，不是和弦准确率。",
        "复训前后四指尖 PCK 指标图",
    )
    add_table(
        document,
        "表 2　复训前后验证结果对比",
        ["指标", "原模型", "复训模型", "变化与解释"],
        [
            ["可见点数 / 验证帧", "356 / 98", "356 / 98", "同一验证集，同一演奏者"],
            ["平均 ROI 归一化误差", "4.897%", "4.623%", "下降 0.274 个百分点"],
            ["中位 ROI 归一化误差", "3.185%", "3.423%", "略有回退，说明改进不是均匀分布"],
            ["PCK@0.05", "72.75%", "73.60%", "提高 0.85 个百分点"],
            ["PCK@0.10", "89.89%", "91.29%", "提高 1.40 个百分点"],
            ["中指 PCK@0.05", "58.16%", "64.29%", "改善最大，但仍为最弱指"],
            ["无名指 PCK@0.05", "74.49%", "69.39%", "出现约 5.1 个百分点回退"],
        ],
        [1.85, 1.1, 1.1, 2.35],
        8.8,
    )
    add_body(
        document,
        "这轮复训的结论应当是“增加数据有效，但不是充分条件”。原始数据的标注框来自人工 ROI，而实时系统使用肤色、"
        "几何和局部跟踪产生的自动 ROI；两种输入分布并不相同。网络在人工框内看到的是较完整的手，自动框可能裁掉拇指、"
        "手腕或部分指尖，也可能包含大量背景。该偏差不会因为继续增加同一场景的图片自动消失。",
    )
    add_h2(document, "4.3 handpose_x 替换试验")
    add_body(
        document,
        "曾尝试用开源的 handpose_x 两阶段流程替换现有手部模型。桌面 NCNN 环境下，手部检测约 14.0 至 19.8 ms/帧，"
        "姿态估计约 24.7 至 33.6 ms/手，但多帧中出现关键点塌缩、落到手外和关节错位。将手框扩成正方形、"
        "按掌心方向旋转到竖直也没有解决；该移植还缺少可用于拒绝错误姿态的置信度。因此没有接入生产链路或部署到板卡。",
    )
    add_figure(
        document,
        "handpose_comparison.png",
        5.75,
        "图 4　同一手部 ROI 下的 handpose_x 与 MediaPipe 对照。左、中为 handpose_x 原方向和竖直方向，右为 MediaPipe 诊断基线；"
        "该图用于说明失败模式，不是生产阈值下的准确率比较。",
        "handpose_x 与 MediaPipe 同 ROI 关键点对照",
    )


def build_section_5(document: Document) -> None:
    add_h1(document, "5 弦品几何、音符投影与音视频融合")
    add_h2(document, "5.1 从指板到物理弦品")
    add_body(
        document,
        "指板区域通过四点单应变换校正到近似矩形坐标。六根弦不采用固定“每份 1/6”的像素假设，而是依据当前可见弦线或"
        "指板上下边界做等分，并保留边界边距；品丝使用可见横向边界建立连续序列。每个指尖在矩形坐标中找到其纵向位置对应的"
        "相邻品丝，再按横向位置匹配最近的弦线。候选保留相邻弦和相邻品的歧义，避免在边界附近强行给出唯一答案。",
    )
    add_figure(
        document,
        "fret_string_overlay.png",
        5.15,
        "图 5　实测弦线与品丝序列的网页叠加示例。绿色/黄色区域表示可见弦品格，白线给出手部或局部 ROI；"
        "该图证明几何叠加链路存在，不代表每个格都已在所有帧中稳定识别。",
        "指板弦线与品丝实测叠加画面",
    )
    add_body(
        document,
        "音符计算采用标准调弦下的 MIDI = 空弦 MIDI + 实际品位，例如第 5 弦第 2 品的 B2 对应 MIDI 47。"
        "品位编号需要用户根据画面中第一根可见边界的真实品数进行一次确认；系统不会把未知的第一条线擅自当成 0 品。"
        "几何代次变化后确认失效，避免把旧标定套到新指板。",
    )
    add_h2(document, "5.2 为什么四个指尖仍不足以直接得到和弦")
    add_body(
        document,
        "按弦手最多有四个指尖，但吉他的发声由六根弦、开放弦、闷音、横按、拨弦范围、实际压力和推弦等共同决定。"
        "视觉给出的只是指尖位置候选，不能证明弦已被压下，也不能单独解释哪几根弦被扫到。即使四指坐标完全正确，"
        "仍可能出现开放弦、同音异弦、部分和弦音被覆盖或多个合理和弦名。",
    )
    add_body(
        document,
        "因此，视觉模块应输出“位置证据和不确定性”，而不是直接输出最终和弦。音频端通过 FFT 功率谱、谐波模板、"
        "音级集合和最低音提供声音证据；融合层再结合时间连续性、把位、可弹奏指型与常用和弦图数据库。"
        "这一路线比将几百个和弦图作为静态分类标签更符合器乐演奏的生成机制，也更适合处理同一和弦的多种把位。",
    )
    add_h2(document, "5.3 音视频融合的设计原则")
    add_body(
        document,
        "视觉结果必须携带源帧身份和年龄，音频结果携带 FFT 快照序号和新鲜度。融合只比较同一时间窗内的新鲜证据；"
        "视觉失效时不沿用旧指尖，音频静音或输入源变化时清除保持状态。实际演示中可显示三类信息：当前音视频候选、"
        "因果平滑后的稳定名称，以及教学用参考和弦图。三者必须在界面上区分，不能把参考指型伪装成本次识别结果。",
    )


def build_section_6(document: Document) -> None:
    add_h1(document, "6 网页可视化与交互")
    add_body(
        document,
        "网页是实验系统的观测窗口，而不是装饰界面。开发过程中实际显示了视频流、琴颈端点线、手部 ROI、弦线、品丝、"
        "指尖点、弦号、品号和候选音符。叠加层与源分析帧绑定：视频已经前进但结果仍是旧帧时，页面应显示等待或过期状态；"
        "这种“不连续”比错误连续更可信。",
    )
    add_body(
        document,
        "可视化还承担调试功能。琴颈线延长到琴头、手部遮挡导致边界内缩、品丝序列跳号、指尖落在弦间等情况都可以直接从画面定位。"
        "后期应在界面上加入当前帧年龄、网络代次、手部 ROI 来源和每个结果的置信度，便于调试时理解系统为何拒绝输出，"
        "而不是只看到“搜索中”三个字。",
    )


def build_section_7(document: Document) -> None:
    add_h1(document, "7 实验设置与结果汇总")
    add_h2(document, "7.1 测试条件")
    add_body(
        document,
        "主要运行平台为 PYNQ-Z2，目标器件 XC7Z020，摄像头 C922，输入格式 YUYV 640×480。模型和网页位于本地网络，"
        "计算机负责视频传输和浏览器显示，板卡负责视觉与音频处理。性能证据包含真实摄像头健康采样、录制帧回放、"
        "合成运动纹理和模块级单元测试，不能将不同测试粒度合并成一个“总通过率”。",
    )
    add_table(
        document,
        "表 3　项目阶段代表性性能测量",
        ["测量项", "结果", "测试条件"],
        [
            ["V22 回放响应率", "4.63 → 6.03 Hz", "同一 45 帧回放，优化后"],
            ["V22 中位往返耗时", "201.67 → 149.72 ms", "网页到板卡到网页，回放"],
            ["V22 板卡处理中位数", "167.94 → 113.42 ms", "不包括完整显示链路"],
            ["实拍相机采集", "7.61 FPS", "启用网页分析，61.4 s，范围 6.79 至 8.88"],
            ["相机单独采集", "13.30 FPS", "网页停止提交分析任务"],
            ["四指尖神经网络一次调用", "1104 ms", "真实录制图、给定琴颈、相机停用"],
            ["独立 PL 四指尖跟踪", "30.54 ms", "合成运动纹理，仅表示跟踪段"],
            ["音符投影", "4.3 至 5.3 ms", "后期优化，5 至 25 条品丝边界"],
            ["音频融合", "0.410 ms", "合成移动与标定测试"],
        ],
        [2.4, 1.4, 2.65],
        8.8,
    )
    add_body(
        document,
        "这些数字揭示了一个常见误区：某个子模块达到 30 ms 并不等于系统达到 30 FPS。相机采集、JPEG 编解码、"
        "网络调度、结果传输、几何校验和浏览器显示会串并联占用时间。若神经网络 1.1 s 才返回一次，前端只能依赖"
        "较短的光流跟踪在中间帧维持位置，而跟踪初始化、遮挡恢复和最终音符输出仍需等待网络证据。",
    )
    add_h2(document, "7.2 已验证与未验证")
    add_table(
        document,
        "表 4　视觉子系统的证据等级",
        ["能力", "已验证内容", "不能据此宣称"],
        [
            ["琴颈候选与跟踪", "真实录制回放、黑帧拒绝、同一请求链路", "独立边界 IoU 或跨场景准确率"],
            ["四指尖网络", "98 帧验证集、356 个可见点、PCK", "真实自动 ROI 下的和弦正确率"],
            ["弦品几何", "实测线、单应、标定和网页叠加", "全帧稳定六弦 22 品覆盖"],
            ["音符投影", "合成几何、B2/MIDI 47 边界测试", "真实按弦压力或完整 voicing"],
            ["音频融合", "新鲜度、序列、融合函数测试", "整首演奏的稳定和弦 F1"],
            ["板卡执行", "模型和部分 PL 卷积实际运行", "完整双模型实时或 30 Hz"],
        ],
        [1.55, 2.8, 2.15],
        8.8,
    )


def build_section_8(document: Document) -> None:
    add_h1(document, "8 失败原因分析")
    add_h2(document, "8.1 为什么看起来简单的颜色跟踪仍需要 1 s 以上")
    add_body(
        document,
        "“颜色跟踪”只是候选生成的一部分。完整请求还包含图像缩放、Lab 转换、边界候选、长线配对、内部颜色支撑、"
        "横向品丝响应、特征提取、光流初始化、RANSAC、当前帧支持和网页传输。首次搜索没有上一帧先验，必须在整幅画面中"
        "遍历多个假设；失败后的同帧恢复又会增加第二轮计算。因此，1.3 s 不是一次颜色阈值判断的时间，而是完整初始化链路的成本。",
    )
    add_h2(document, "8.2 模型精度不高的结构性原因")
    add_body(
        document,
        "第一，训练标签来自 1920×1080 视频，而实时板卡使用 640×480，视角、压缩、模糊和手部尺度均发生变化。"
        "第二，人工 ROI 与自动 ROI 的裁剪分布不同，网络在训练时看到较完整的手，运行时可能只看到局部。"
        "第三，四热图没有显式可见性标签和拒绝机制，遮挡或抬指时仍可能给出峰值。"
        "第四，逐帧独立取峰值会产生跳点，未充分利用手指运动学和时间连续性。第五，单演奏者、单环境数据无法验证跨人泛化。",
    )
    add_h2(document, "8.3 琴颈跟踪丢失的机制")
    add_body(
        document,
        "光流假设局部像素外观在相邻帧近似保持。快速运动造成大位移、运动模糊和曝光变化，光流容易落到重复的木纹、品丝或手指纹理上。"
        "手部遮挡会暂时改变琴颈外观，使特征点分布偏移；当 RANSAC 把部分手指纹理当成琴颈边界时，四边形会变形。"
        "端点式表示减少了中间遮挡的影响，但远端点仍可能落在琴枕、琴头或背景边缘。工程上应结合多帧运动模型、"
        "琴颈宽度和边界方向的先验，而不是只依赖当前帧特征数量。",
    )
    add_h2(document, "8.4 弦品几何和和弦级验收缺失")
    add_body(
        document,
        "当前弦品线多为可见线检测的结果，缺少与人工标定逐线对应的真值。没有独立真值，就只能统计“是否输出候选”，"
        "不能统计弦号、品号与边界位置的准确率。四个指尖也不足以表达开放弦、闷音、横按、拨弦范围和压力。"
        "最终和弦还需要音频、指型约束和时间平滑共同参与。项目后期把有限时间用于继续调试单个视觉模块，未完成这一层的实验设计。",
    )
    add_h2(document, "8.5 板卡加速的物理边界")
    add_body(
        document,
        "现有完整设计已使用 212/220 DSP、125/140 BRAM36 和 36,446/53,200 LUT。把第二套未修改 YOLO 直接加入 PL 不可行。"
        "离线预优化提出了 W8A16 共享卷积核，隔离综合约为 82 DSP、49 BRAM36、20,422 LUT，资源算术上可留下约"
        "67 DSP、15 BRAM36 和 6,274 LUT；但这是未布线的可行性估计，不包含完整图执行、互连、DDR 和调度开销。"
        "项目因此没有把“双模型都放进 PL”写成已完成成果。",
    )


def build_section_9(document: Document) -> None:
    add_h1(document, "9 技术路线评价与关键决策")
    add_h2(document, "9.1 为什么没有继续使用 YOLO 作为主琴颈方案")
    add_body(
        document,
        "YOLO 是有效的目标检测路线，但项目缺少专门的琴颈训练集和独立测试集。通用预训练模型可能检测“人”或“吉他”，"
        "却不一定能稳定给出指板边界；在 PS 上曾观察到一次约 7.95 s 的推理和低于阈值的置信度。"
        "此外，YOLO 320×320 版本的卷积 MAC 约为四指尖模型的 10.3 倍，完整图还需要 SiLU、Sigmoid、Concat、"
        "Resize、解码和 NMS，现有 PL 编译器并不支持完整图。没有数据与完整加速闭环时，YOLO 不能自动解决几何问题。",
    )
    add_body(
        document,
        "更合理的组合是“低成本连续跟踪 + 小目标检测或颜色候选的周期性重获”。琴颈是大尺度、长条状目标，"
        "端点、长边和透视约束是强先验；YOLO 可作为首次锁定和长期丢失后的重获器，而不是每帧都运行。"
        "这既能利用学习模型的语义鲁棒性，又能把实时更新交给光流和几何模型。",
    )
    add_h2(document, "9.2 为什么 handpose_x 没有替换现有模型")
    add_body(
        document,
        "替换模型的首要标准不是桌面速度，而是同 ROI 下的关键点稳定性和失败可检测性。handpose_x 在五帧测试中能检出单只手，"
        "但多个手指关键点塌缩或落到手外，且移植缺少姿态置信度。MediaPipe 虽然也有遮挡失败，但至少可以拒绝低置信度结果。"
        "因此本次没有采用 handpose_x。该试验的价值在于明确了一个研究要求：新模型必须同时提供点位置、可见性和可校准的置信度。",
    )
    add_h2(document, "9.3 项目已形成的工程特点")
    add_body(
        document,
        "第一，数据身份隔离严格：旧网络结果、旧几何和旧音频不会自动混入当前帧。第二，视觉表示可解释：琴颈端点、单应、"
        "弦线和品丝都能在网页上直接检查。第三，模块分工明确：PS 做几何和状态，PL 做规则图像与卷积热点。"
        "第四，测量记录保留了测试条件和限制，没有把离线结果包装成现场准确率。这些特点使得项目即使未完成，"
        "仍可作为后续研究的基础设施和数据采集平台。",
    )


def build_section_10(document: Document) -> None:
    add_h1(document, "10 后续研究计划")
    add_body(
        document,
        "后续工作应从“继续堆叠功能”转为四个相互约束的验证问题："
        "（1）如何在遮挡和快速运动下获得稳定的琴颈/指板几何；（2）如何在自动 ROI 下同时估计指尖位置和可见性；"
        "（3）如何建立弦品真值与和弦级评估；（4）如何在 PYNQ-Z2 上做延迟感知的 PS/PL 调度。"
        "这四部分共同决定系统能否从演示原型转为可测量研究平台。",
    )
    add_h2(document, "10.1 第一阶段：数据与评测协议")
    add_body(
        document,
        "用最终 C922 和板卡采集链路建立跨演奏者、跨光照、跨距离的数据集，至少包含 3 名演奏者、2 台设备或两种拍摄距离、"
        "稳定按弦、换把、遮挡、抬指和重新进入。按整段录像而非相邻帧划分训练、验证和独立测试，确保同一演奏片段不泄漏。"
        "每帧同时记录人工 ROI、自动 ROI、四指尖可见性、弦号、品号和完整发声和弦。",
    )
    add_body(
        document,
        "最小可用验收指标包括：琴颈边界 IoU 与端点误差，指尖 PCK、可见性 F1、时间抖动，弦号准确率、品号误差，"
        "以及和弦 macro-F1、Top-3 准确率和音频静音时的误报率。单独看平均像素误差不足以说明系统能否映射到正确音符。",
    )
    add_h2(document, "10.2 第二阶段：可见性感知的手部模型")
    add_body(
        document,
        "保留现有四指尖热图作为可解释基线，但增加手指可见性头或独立手部/指尖分割头，让模型能输出“不可见”或“低置信度”。"
        "训练时同一源帧分别输入人工框与自动框，比较两种 ROI 下的误差，区分裁框误差和网络误差。"
        "运行时结合光流和手部仿射运动做时间平滑，但只在局部跟踪质量满足阈值时使用。若精度仍不足，再考虑 21 点姿态先验、"
        "轻量 Transformer 或蒸馏学生模型，而不直接替换为缺少置信度的黑盒移植。",
    )
    add_h2(document, "10.3 第三阶段：弦品真值与音视频融合")
    add_body(
        document,
        "建立弦线和品丝的逐帧人工标注，记录可见边界、遮挡区间和第一根可见品丝的真实编号。把指尖输出表示为离散弦品候选和"
        "连续坐标，允许相邻弦、相邻品的多峰分布。音频端保留谐波模板与最低音约束，融合层输出候选和弦的分数分布，"
        "再用时间状态机平滑。和弦参考图作为教学显示，与本次识别结果分开。",
    )
    add_h2(document, "10.4 第四阶段：延迟感知的 FPGA 协同")
    add_body(
        document,
        "硬件侧优先实现共享 W8A16 卷积核、权重分块加载、层融合和片上复用，减少 DDR 往返。保持现有 Q16 网络作为精度参考，"
        "任何低位结果都必须在同一测试集上与浮点参考配对比较。调度上把琴颈跟踪、手部检测、指尖网络和弦品几何按不同频率运行，"
        "使用有界队列和结果年龄拒绝过期输出。",
    )
    add_table(
        document,
        "表 5　建议的阶段目标与验收方式",
        ["阶段", "研究目标", "可验收结果", "暂定门槛"],
        [
            ["3 个月", "自动 ROI 与可见性基线", "跨录像独立测试、PCK 与可见性 F1", "建立可重复评测脚本"],
            ["6 个月", "稳定琴颈与弦品几何", "端点误差、弦号线匹配、顺序错误统计", "独立视频上持续可用"],
            ["9 个月", "音视频融合和弦", "macro-F1、Top-3、换和弦延迟", "常见和弦可解释闭环"],
            ["12 个月", "共享 PL 加速", "完整图延迟、p95、资源与时序", "NN ≤ 100 ms，整链 ≤ 150 ms"],
        ],
        [0.85, 1.65, 2.45, 1.7],
        8.7,
    )
    add_body(
        document,
        "上述门槛是研究目标，不是已经达到的指标。若 30 FPS 在现有器件上仍不可达，可以先用 10 至 15 FPS 的稳定视觉更新，"
        "配合更高频的局部跟踪与音频更新，把“视觉事实更新”和“画面刷新”分开评估。这样的系统仍然可能对交互式教学有价值，"
        "但需要在论文中明确延迟和适用边界。",
    )


def build_section_11(document: Document) -> None:
    add_h1(document, "11 可复现性与已知风险")
    add_body(
        document,
        "工程证据与可复现文件已发布到 GitHub 仓库：",
        first_indent=False,
    )
    repository_paragraph = document.add_paragraph()
    repository_paragraph.paragraph_format.space_after = Pt(6)
    add_hyperlink(repository_paragraph, REPOSITORY_URL, REPOSITORY_URL, 9.4)
    add_body(
        document,
        "下列链接指向仓库中的琴颈跟踪、颜色边界、神经网络加速、双模型预优化、指尖数据采集、音符投影和复训指标文件。",
        first_indent=False,
    )
    evidence = [
        ("琴颈跟踪", r"reports\NECK_TRACKING_20261027.md"),
        ("颜色边界与恢复", r"reports\NECK_COLOR_TRACKING_20261028.md"),
        ("板卡指尖加速", r"reports\NEURAL_ACCELERATION_20261027.md"),
        ("低比特与双模型可行性", r"reports\DUAL_NEURAL_PREOPT_20261028.md"),
        ("指尖数据与评测安排", r"reports\FINGERTIP_DATA_COLLECTION_20261028.md"),
        ("音符投影", r"reports\FINGERTIP_NOTES_20261027.md"),
        ("复训指标", r"output\releases\fingertip_reviewed_20261030\evidence\metrics.json"),
        ("handpose_x 对照", r"experiments\handpose_x_trial\README.md"),
    ]
    for label, relative in evidence:
        p = document.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(f"{label}：")
        set_run_font(run, 9.9, True)
        repository_path = relative.replace("\\", "/")
        add_hyperlink(
            p,
            f"GitHub / {repository_path}",
            f"{REPOSITORY_URL}/blob/main/{repository_path}",
            9.2,
        )
    add_body(
        document,
        "已知风险包括：摄像头 USB 在异常状态下可能阻塞，普通服务重启不一定恢复；板卡运行目录与开发目录可能包含不同版本；"
        "部分早期报告描述的是历史状态，应与 10 月 30 日之后的记录区分；网页端口、Python/OpenCV 版本和本地网络地址会影响复现。"
        "恢复工程前应校验 BIT/HWH、模型权重、运行目录和服务状态，而不是直接覆盖正在运行的系统。",
    )
    add_body(
        document,
        "本报告使用的最新复训模型已经保存为 best.pt，但指标文件明确标记 pl_deployment_verified 为 false。"
        "因此，报告中的 73.60% 是离线/软件验证结果，不应描述成板卡网页已经达到的实时精度。"
        "板卡端曾经运行的是已验证的既有网络与跟踪链路，后续若替换权重，仍需要重新做伪量化、端到端回放和真实演奏验收。",
    )


def build_section_12(document: Document) -> None:
    add_h1(document, "12 结论")
    add_body(
        document,
        "这次项目没有在比赛截止前完成实时、稳定的视觉和弦识别，主要原因是视觉几何、手部关键点、自动 ROI、"
        "弦品真值和 FPGA 端到端延迟共同形成的系统性问题。结果不能简单归结为“模型不够大”或“图片不够多”："
        "复训说明数据有帮助，但提升幅度有限，结构偏差依然存在。",
    )
    add_body(
        document,
        "项目仍然交付了可验证的基础：真实相机与板卡链路、端点式琴颈跟踪、四指尖热图网络、弦品几何、网页叠加、"
        "音符投影、新鲜度隔离和一组可追溯的性能记录。它证明了“视觉位置 + 音频证据”的研究方向具有工程可行性，"
        "也清楚暴露了下一步必须解决的实验问题。",
    )
    add_body(
        document,
        "后续工作应在这套基础上把目标收敛为三个可交付成果：一个带自动 ROI 和可见性标注的指尖数据集，"
        "一个可在独立视频上验收的琴颈与弦品几何模型，以及一个带延迟分布和和弦级指标的 PS/PL 协同原型。"
        "如果这些条件能够建立，项目将从比赛演示推进为可复现的实验系统。",
    )


def build_appendix(document: Document) -> None:
    add_h1(document, "附录 A　指标定义与阅读说明", page_break=True)
    add_h2(document, "A.1 PCK")
    add_body(
        document,
        "本报告使用 ROI 归一化距离定义 PCK。对预测点 (x,y) 与真值点 (x*,y*)，"
        "d = sqrt(((x-x*)/W_roi)^2 + ((y-y*)/H_roi)^2)。当 d ≤ 0.05 时记为 PCK@0.05，"
        "当 d ≤ 0.10 时记为 PCK@0.10。该指标只衡量指尖点距离，不衡量手指身份交换是否会改变弦号，"
        "也不衡量压力和实际发声。",
    )
    add_h2(document, "A.2 FPS、Hz 与延迟")
    add_body(
        document,
        "相机 FPS 表示成功取到新图像的频率；分析 Hz 表示前端成功提交或收到分析结果的频率；请求中位数表示单次请求往返的时间尺度；"
        "p95 表示尾部延迟。三者不能互相替代。一个系统可能以 13 FPS 显示视频，但神经网络每 1.1 s 才更新一次；"
        "也可能每个模块都很快，却因队列和同步产生较大端到端 p95。",
    )
    add_h2(document, "A.3 证据等级")
    add_body(
        document,
        "本报告按四类证据引用结果：真实板卡运行与健康采样；录制视频通过真实板卡链路的回放；受控合成输入上的模块测试；"
        "离线模型验证。合成测试用于验证几何和协议，不用于证明真实演奏准确率。报告没有把不同粒度的测试数量相加为总通过率。",
    )
    add_h1(document, "参考与进一步阅读")
    refs = [
        "[1] AMD. Zynq-7000 SoC Technical Reference Manual, UG585.",
        "[2] PYNQ Project. PYNQ-Z2 board documentation and PYNQ Overlay guides.",
        "[3] Zhang F. et al. MediaPipe Hands: On-device Real-time Hand Tracking. arXiv:2006.10214, 2020.",
        "[4] MMPose Contributors. MMPose: OpenMMLab pose estimation toolbox. https://github.com/open-mmlab/mmpose",
        "[5] Google AI Edge. MediaPipe Hand Landmarker documentation. https://ai.google.dev/edge/mediapipe",
        "[6] EricLee. handpose_x open-source project. https://codechina.csdn.net/EricLee/handpose_x",
        "[7] FINN project. FINN HLSLib dataflow neural-network operators. https://github.com/Xilinx/finn-hlslib",
        "[8] hls4ml project. https://github.com/fastmachinelearning/hls4ml",
        "[9] Hartley R., Zisserman A. Multiple View Geometry in Computer Vision. Cambridge University Press, 2004.",
        "[10] Bouguet J.-Y. Pyramidal Implementation of the Lucas-Kanade Feature Tracker. Intel technical report, 2001.",
    ]
    for ref in refs:
        p = document.add_paragraph()
        p.paragraph_format.left_indent = Inches(0)
        p.paragraph_format.first_line_indent = Inches(-0.16)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(ref)
        set_run_font(run, 9.2)
    p = document.add_paragraph()
    p.paragraph_format.left_indent = Inches(0)
    p.paragraph_format.first_line_indent = Inches(-0.16)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("[11] 项目公开证据：")
    set_run_font(run, 9.2)
    public_evidence = [
        "reports/NEURAL_ACCELERATION_20261027.md",
        "reports/NECK_COLOR_TRACKING_20261028.md",
        "reports/FINGERTIP_NOTES_20261027.md",
        "output/releases/fingertip_reviewed_20261030/evidence/metrics.json",
    ]
    for index, relative in enumerate(public_evidence):
        if index:
            separator = p.add_run("、")
            set_run_font(separator, 9.2)
        add_hyperlink(
            p,
            relative,
            f"{REPOSITORY_URL}/blob/main/{relative}",
            9.2,
        )
    end = p.add_run("。")
    set_run_font(end, 9.2)


def build() -> None:
    create_assets()
    document = configure_document()
    build_cover(document)
    build_summary(document)
    build_section_1(document)
    build_section_2(document)
    build_section_3(document)
    build_section_4(document)
    build_section_5(document)
    build_section_6(document)
    build_section_7(document)
    build_section_8(document)
    build_section_9(document)
    build_section_10(document)
    build_section_11(document)
    build_section_12(document)
    build_appendix(document)
    document.save(OUTPUT)

    manifest = {
        "output": str(OUTPUT),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "source_files": [
            str(PROJECT / r"reports\NECK_TRACKING_20261027.md"),
            str(PROJECT / r"reports\NECK_COLOR_TRACKING_20261028.md"),
            str(PROJECT / r"reports\NEURAL_ACCELERATION_20261027.md"),
            str(PROJECT / r"reports\DUAL_NEURAL_PREOPT_20261028.md"),
            str(PROJECT / r"reports\FINGERTIP_DATA_COLLECTION_20261028.md"),
            str(PROJECT / r"reports\FINGERTIP_NOTES_20261027.md"),
            str(PROJECT / r"output\releases\fingertip_reviewed_20261030\evidence\metrics.json"),
            str(PROJECT / r"experiments\handpose_x_trial\README.md"),
        ],
        "figures": [
            str(ASSETS / "vision_pipeline.png"),
            str(ASSETS / "endpoint_tracking.png"),
            str(ASSETS / "fret_string_overlay.png"),
            str(ASSETS / "fingertip_metrics.png"),
            str(ASSETS / "handpose_comparison.png"),
        ],
        "render_review_required": True,
        "notes": "Word is the primary deliverable; PDF is rendered from the final DOCX for review and delivery.",
    }
    (ROOT / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
