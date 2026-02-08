from __future__ import annotations

import io
import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, field_validator

from art_director.utils import image_bytes_to_b64


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _parse_hex_color(value: str, *, alpha: int = 255) -> tuple[int, int, int, int]:
    s = (value or "").strip()
    if not s:
        return (0, 0, 0, alpha)
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return (0, 0, 0, alpha)
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError:
        return (0, 0, 0, alpha)
    a = alpha
    if a < 0:
        a = 0
    if a > 255:
        a = 255
    return (r, g, b, a)


def _load_font(font_family: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(font_family, size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("arial", size)
    except Exception:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def _wrap_text(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str, max_width: float) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    lines: list[str] = []
    current: list[str] = []

    for w in words:
        trial = " ".join([*current, w])
        w_px, _ = _text_size(draw, trial, font)
        if current and w_px > max_width:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)

    if current:
        lines.append(" ".join(current))
    return lines


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    dash_len: float,
    gap_len: float,
    width: int,
    fill: tuple[int, int, int, int],
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist <= 0.001:
        return

    ux = dx / dist
    uy = dy / dist
    pos = 0.0
    while pos < dist:
        seg_start = pos
        seg_end = min(dist, pos + dash_len)
        sx = x1 + ux * seg_start
        sy = y1 + uy * seg_start
        ex = x1 + ux * seg_end
        ey = y1 + uy * seg_end
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        pos += dash_len + gap_len


def _arrowhead_points(
    tip: tuple[float, float],
    tail: tuple[float, float],
    *,
    length: float,
    width: float,
) -> list[tuple[float, float]]:
    tx, ty = tip
    sx, sy = tail
    dx = tx - sx
    dy = ty - sy
    dist = math.hypot(dx, dy)
    if dist <= 0.001:
        return [(tx, ty), (tx, ty), (tx, ty)]
    ux = dx / dist
    uy = dy / dist
    bx = tx - ux * length
    by = ty - uy * length
    px = -uy
    py = ux
    left = (bx + px * width * 0.5, by + py * width * 0.5)
    right = (bx - px * width * 0.5, by - py * width * 0.5)
    return [(tx, ty), left, right]


def _clip_line_to_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    rect_center: tuple[float, float],
    rect_w: float,
    rect_h: float,
    *,
    reverse: bool,
) -> tuple[float, float]:
    cx, cy = rect_center
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    if reverse:
        dx = -dx
        dy = -dy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return start

    adx = abs(dx)
    ady = abs(dy)
    half_w = rect_w * 0.5
    half_h = rect_h * 0.5
    if adx < 1e-6:
        t = half_h / ady
    elif ady < 1e-6:
        t = half_w / adx
    else:
        t = min(half_w / adx, half_h / ady)

    ox = dx * t
    oy = dy * t
    if reverse:
        return (cx - ox, cy - oy)
    return (cx + ox, cy + oy)


class DiagramNode(BaseModel):
    id: str
    label: str
    sublabel: str = ""
    x: float = Field(default=0.5, ge=0.0, le=1.0)
    y: float = Field(default=0.5, ge=0.0, le=1.0)
    width: float = 0.18
    height: float = 0.12
    color: str = "#1f2937"
    text_color: str = "#ffffff"
    border_color: str = "#374151"
    shape: str = "rounded_rect"
    group: str = ""

    @field_validator("shape")
    @classmethod
    def _shape_valid(cls, v: str) -> str:
        allowed = {"rounded_rect", "rect", "diamond", "ellipse", "hexagon"}
        if v not in allowed:
            return "rounded_rect"
        return v


class DiagramEdge(BaseModel):
    source: str
    target: str
    label: str = ""
    style: str = "solid"
    color: str = "#9ca3af"
    arrow: bool = True

    @field_validator("style")
    @classmethod
    def _style_valid(cls, v: str) -> str:
        if v not in {"solid", "dashed"}:
            return "solid"
        return v


class DiagramSpec(BaseModel):
    title: str = ""
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    groups: dict[str, Any] | list[Any] = Field(default_factory=dict)
    width: int = 1024
    height: int = 768
    background_color: str = "#0f172a"
    font_family: str = "arial"

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        if not data.get("groups"):
            data.pop("groups", None)
        return data

    @field_validator("groups", mode="before")
    @classmethod
    def _normalize_groups(cls, v: Any) -> dict[str, dict[str, str]]:
        if v is None:
            return {}

        if isinstance(v, dict):
            out: dict[str, dict[str, str]] = {}
            for k, raw in v.items():
                if isinstance(raw, dict):
                    label = str(raw.get("label", k))
                    color = str(raw.get("color", "#334155"))
                else:
                    label = str(k)
                    color = "#334155"
                out[str(k)] = {"label": label, "color": color}
            return out

        if isinstance(v, list):
            out2: dict[str, dict[str, str]] = {}
            for item in v:
                if not isinstance(item, dict):
                    continue
                gid = item.get("id")
                if not gid:
                    continue
                label = str(item.get("label", gid))
                color = str(item.get("color", "#334155"))
                out2[str(gid)] = {"label": label, "color": color}
            return out2

        return {}

    @field_validator("width")
    @classmethod
    def _width_positive(cls, v: int) -> int:
        return v if v > 0 else 1

    @field_validator("height")
    @classmethod
    def _height_positive(cls, v: int) -> int:
        return v if v > 0 else 1


class DiagramRenderer:
    def render(self, spec: DiagramSpec) -> bytes:
        scale = 2
        w2 = int(spec.width * scale)
        h2 = int(spec.height * scale)

        img = Image.new("RGBA", (w2, h2), _parse_hex_color(spec.background_color, alpha=255))
        draw = ImageDraw.Draw(img, "RGBA")

        fonts = {
            "title": _load_font(spec.font_family, 48),
            "label": _load_font(spec.font_family, 28),
            "sublabel": _load_font(spec.font_family, 22),
            "edge": _load_font(spec.font_family, 20),
        }

        node_boxes: dict[str, tuple[float, float, float, float]] = {}
        node_centers: dict[str, tuple[float, float]] = {}
        for n in spec.nodes:
            cx = _clamp01(n.x) * w2
            cy = _clamp01(n.y) * h2
            nw = max(8.0, n.width * w2)
            nh = max(8.0, n.height * h2)
            x0 = cx - nw * 0.5
            y0 = cy - nh * 0.5
            x1 = cx + nw * 0.5
            y1 = cy + nh * 0.5
            node_boxes[n.id] = (x0, y0, x1, y1)
            node_centers[n.id] = (cx, cy)

        self._draw_groups(draw, spec, node_boxes, fonts)
        self._draw_edges(draw, spec, node_boxes, node_centers, fonts)
        self._draw_nodes(draw, spec, node_boxes, fonts)
        self._draw_title(draw, spec, fonts)

        out = img.resize((spec.width, spec.height), resample=Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    def render_to_b64(self, spec: DiagramSpec) -> str:
        data = self.render(spec)
        return image_bytes_to_b64(data, mime="image/png")

    def _draw_groups(
        self,
        draw: ImageDraw.ImageDraw,
        spec: DiagramSpec,
        node_boxes: dict[str, tuple[float, float, float, float]],
        fonts: dict[str, ImageFont.ImageFont],
    ) -> None:
        if not isinstance(spec.groups, dict):
            return

        by_group: dict[str, list[str]] = {}
        for n in spec.nodes:
            if n.group:
                by_group.setdefault(n.group, []).append(n.id)

        pad = 36
        label_pad = 34
        for gid, node_ids in by_group.items():
            if not node_ids:
                continue
            boxes = [node_boxes[nid] for nid in node_ids if nid in node_boxes]
            if not boxes:
                continue
            x0 = min(b[0] for b in boxes) - pad
            y0 = min(b[1] for b in boxes) - pad - label_pad
            x1 = max(b[2] for b in boxes) + pad
            y1 = max(b[3] for b in boxes) + pad

            gmeta = spec.groups.get(gid) if isinstance(spec.groups, dict) else None
            if isinstance(gmeta, dict):
                label = str(gmeta.get("label", gid))
                color = str(gmeta.get("color", "#334155"))
            else:
                label = gid
                color = "#334155"

            fill = _parse_hex_color(color, alpha=35)
            outline = _parse_hex_color(color, alpha=90)
            radius = int(min((x1 - x0), (y1 - y0)) * 0.08)
            draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=2)

            if label:
                tw, th = _text_size(draw, label, fonts["edge"])
                tx = x0 + 16
                ty = y0 + 12
                bg = (15, 23, 42, 150)
                draw.rounded_rectangle([tx - 8, ty - 6, tx + tw + 8, ty + th + 6], radius=10, fill=bg)
                draw.text((tx, ty), label, font=fonts["edge"], fill=_parse_hex_color("#e5e7eb", alpha=230))

    def _draw_edges(
        self,
        draw: ImageDraw.ImageDraw,
        spec: DiagramSpec,
        node_boxes: dict[str, tuple[float, float, float, float]],
        node_centers: dict[str, tuple[float, float]],
        fonts: dict[str, ImageFont.ImageFont],
    ) -> None:
        for e in spec.edges:
            if e.source not in node_centers or e.target not in node_centers:
                continue

            sx, sy = node_centers[e.source]
            tx, ty = node_centers[e.target]
            sb = node_boxes.get(e.source)
            tb = node_boxes.get(e.target)
            if not sb or not tb:
                continue

            sw = sb[2] - sb[0]
            sh = sb[3] - sb[1]
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]

            start = _clip_line_to_rect((sx, sy), (tx, ty), (sx, sy), sw, sh, reverse=False)
            end = _clip_line_to_rect((sx, sy), (tx, ty), (tx, ty), tw, th, reverse=True)

            stroke = _parse_hex_color(e.color, alpha=220)
            width = 4

            if e.style == "dashed":
                _draw_dashed_line(draw, start, end, dash_len=18, gap_len=10, width=width, fill=stroke)
            else:
                draw.line([start, end], fill=stroke, width=width)

            if e.arrow:
                pts = _arrowhead_points(end, start, length=22, width=18)
                draw.polygon(pts, fill=stroke)

            if e.label:
                mx = (start[0] + end[0]) * 0.5
                my = (start[1] + end[1]) * 0.5
                label = e.label.strip()
                tw_px, th_px = _text_size(draw, label, fonts["edge"])
                pad = 8
                bg = (2, 6, 23, 170)
                box = [mx - tw_px / 2 - pad, my - th_px / 2 - pad, mx + tw_px / 2 + pad, my + th_px / 2 + pad]
                draw.rounded_rectangle(box, radius=10, fill=bg)
                draw.text((mx - tw_px / 2, my - th_px / 2), label, font=fonts["edge"], fill=(255, 255, 255, 230))

    def _draw_nodes(
        self,
        draw: ImageDraw.ImageDraw,
        spec: DiagramSpec,
        node_boxes: dict[str, tuple[float, float, float, float]],
        fonts: dict[str, ImageFont.ImageFont],
    ) -> None:
        for n in spec.nodes:
            if n.id not in node_boxes:
                continue
            x0, y0, x1, y1 = node_boxes[n.id]
            fill = _parse_hex_color(n.color, alpha=255)
            outline = _parse_hex_color(n.border_color, alpha=255)
            text_fill = _parse_hex_color(n.text_color, alpha=255)

            shadow_fill = (0, 0, 0, 95)
            self._draw_shape(draw, n.shape, (x0 + 4, y0 + 4, x1 + 4, y1 + 4), fill=shadow_fill, outline=None)
            self._draw_shape(draw, n.shape, (x0, y0, x1, y1), fill=fill, outline=outline)

            self._draw_node_text(draw, n, (x0, y0, x1, y1), fonts, text_fill)

    def _draw_shape(
        self,
        draw: ImageDraw.ImageDraw,
        shape: str,
        box: tuple[float, float, float, float],
        *,
        fill: tuple[int, int, int, int],
        outline: tuple[int, int, int, int] | None,
    ) -> None:
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0

        if shape == "rounded_rect":
            radius = int(min(w, h) * 0.15)
            draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=4)
            return
        if shape == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=4)
            return
        if shape == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=fill, outline=outline, width=4)
            return
        if shape == "diamond":
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            pts = [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
            draw.polygon(pts, fill=fill, outline=outline)
            if outline is not None:
                draw.line([pts[0], pts[1], pts[2], pts[3], pts[0]], fill=outline, width=4)
            return
        if shape == "hexagon":
            dx = w * 0.25
            pts2 = [
                (x0 + dx, y0),
                (x1 - dx, y0),
                (x1, (y0 + y1) * 0.5),
                (x1 - dx, y1),
                (x0 + dx, y1),
                (x0, (y0 + y1) * 0.5),
            ]
            draw.polygon(pts2, fill=fill, outline=outline)
            if outline is not None:
                draw.line([*pts2, pts2[0]], fill=outline, width=4)
            return
        radius2 = int(min(w, h) * 0.15)
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius2, fill=fill, outline=outline, width=4)

    def _draw_node_text(
        self,
        draw: ImageDraw.ImageDraw,
        node: DiagramNode,
        box: tuple[float, float, float, float],
        fonts: dict[str, ImageFont.ImageFont],
        color: tuple[int, int, int, int],
    ) -> None:
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5

        max_text_w = w * 0.82
        label_lines = _wrap_text(draw, fonts["label"], node.label, max_text_w)
        sub_lines = _wrap_text(draw, fonts["sublabel"], node.sublabel, max_text_w)

        def _line_height(font: ImageFont.ImageFont) -> int:
            _, th = _text_size(draw, "Ag", font)
            return th

        lh = _line_height(fonts["label"])
        sh = _line_height(fonts["sublabel"])
        label_block = lh * len(label_lines)
        sub_block = sh * len(sub_lines)
        gap = 10 if (label_lines and sub_lines) else 0
        total_h = label_block + gap + sub_block

        y = cy - total_h * 0.5
        for line in label_lines:
            tw, _ = _text_size(draw, line, fonts["label"])
            draw.text((cx - tw * 0.5, y), line, font=fonts["label"], fill=color)
            y += lh
        if gap:
            y += gap
        for line in sub_lines:
            tw, _ = _text_size(draw, line, fonts["sublabel"])
            draw.text((cx - tw * 0.5, y), line, font=fonts["sublabel"], fill=(color[0], color[1], color[2], 220))
            y += sh

        if node.shape == "diamond" and node.sublabel and total_h > h * 0.75:
            return

    def _draw_title(
        self,
        draw: ImageDraw.ImageDraw,
        spec: DiagramSpec,
        fonts: dict[str, ImageFont.ImageFont],
    ) -> None:
        title = (spec.title or "").strip()
        if not title:
            return
        tw, th = _text_size(draw, title, fonts["title"])
        x = (spec.width * 2 - tw) * 0.5
        y = 18
        shadow = (0, 0, 0, 140)
        draw.text((x + 2, y + 2), title, font=fonts["title"], fill=shadow)
        draw.text((x, y), title, font=fonts["title"], fill=_parse_hex_color("#e5e7eb", alpha=245))
