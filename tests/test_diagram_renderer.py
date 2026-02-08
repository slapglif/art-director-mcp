from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image
from pydantic import ValidationError

from art_director.diagram_renderer import (
    DiagramEdge,
    DiagramNode,
    DiagramRenderer,
    DiagramSpec,
)


def _assert_png_bytes(data: bytes, *, min_size: int = 1) -> tuple[int, int]:
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > min_size
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(BytesIO(data))
    img.load()
    return img.size


def _simple_spec(*, width: int = 640, height: int = 360, title: str = "") -> DiagramSpec:
    return DiagramSpec(title=title, width=width, height=height)


def test_diagram_spec_defaults() -> None:
    spec = DiagramSpec()
    assert isinstance(spec.nodes, list)
    assert isinstance(spec.edges, list)
    assert spec.width > 0
    assert spec.height > 0
    assert isinstance(spec.background_color, str)
    assert spec.background_color
    assert isinstance(spec.font_family, str)
    assert spec.font_family


def test_diagram_node_validation() -> None:
    try:
        node = DiagramNode(
            id="n1",
            label="Node",
            x=-0.25,
            y=1.25,
        )
    except ValidationError:
        return
    else:
        assert 0.0 <= node.x <= 1.0
        assert 0.0 <= node.y <= 1.0


def test_render_empty_spec() -> None:
    renderer = DiagramRenderer()
    data = renderer.render(_simple_spec())
    _assert_png_bytes(data)


def test_render_single_node() -> None:
    renderer = DiagramRenderer()
    spec = _simple_spec()
    spec.nodes.append(
        DiagramNode(
            id="core",
            label="Core",
            sublabel="art_director",
            x=0.5,
            y=0.5,
            width=0.28,
            height=0.18,
            color="#1f2937",
            text_color="#ffffff",
            border_color="#0b1020",
            shape="rounded_rect",
        )
    )
    data = renderer.render(spec)
    _assert_png_bytes(data)


def test_render_with_edges() -> None:
    renderer = DiagramRenderer()
    spec = _simple_spec(width=900, height=420)
    spec.nodes.extend(
        [
            DiagramNode(id="planner", label="Planner", x=0.25, y=0.5, shape="rounded_rect"),
            DiagramNode(id="executor", label="Executor", x=0.75, y=0.5, shape="rounded_rect"),
        ]
    )
    spec.edges.append(DiagramEdge(source="planner", target="executor", label="plan"))
    data = renderer.render(spec)
    _assert_png_bytes(data)


@pytest.mark.parametrize("shape", ["rounded_rect", "rect", "diamond", "ellipse", "hexagon"])
def test_render_all_shapes(shape: str) -> None:
    renderer = DiagramRenderer()
    spec = _simple_spec(width=700, height=360)
    spec.nodes.append(
        DiagramNode(
            id=f"s-{shape}",
            label=shape,
            x=0.5,
            y=0.5,
            width=0.4,
            height=0.22,
            shape=shape,
            color="#0ea5e9",
            text_color="#0b1020",
            border_color="#075985",
        )
    )
    data = renderer.render(spec)
    _assert_png_bytes(data)


def test_render_with_title() -> None:
    renderer = DiagramRenderer()
    spec = _simple_spec(title="Art Director MCP")
    data = renderer.render(spec)
    _assert_png_bytes(data)


def test_render_with_groups() -> None:
    renderer = DiagramRenderer()
    spec = _simple_spec(width=1000, height=600, title="Grouped Diagram")

    nodes = [
        DiagramNode(id="cfg", label="config.py", x=0.2, y=0.35, group="core"),
        DiagramNode(id="sch", label="schemas.py", x=0.2, y=0.65, group="core"),
        DiagramNode(id="srv", label="server.py", x=0.8, y=0.5, group="mcp"),
    ]
    spec.nodes.extend(nodes)
    spec.edges.append(DiagramEdge(source="srv", target="cfg", label="uses"))

    candidate_groups: list[object] = [
        {
            "core": {"label": "Core", "color": "#f59e0b"},
            "mcp": {"label": "MCP", "color": "#22c55e"},
        },
        [
            {"id": "core", "label": "Core", "color": "#f59e0b"},
            {"id": "mcp", "label": "MCP", "color": "#22c55e"},
        ],
    ]

    last_exc: Exception | None = None
    for groups in candidate_groups:
        try:
            spec2 = DiagramSpec(**spec.model_dump(), groups=groups)
            data = renderer.render(spec2)
            _assert_png_bytes(data)
            return
        except Exception as exc:
            last_exc = exc

    raise AssertionError(f"Rendering with groups failed. Last error: {last_exc}")


def test_render_dashed_edges() -> None:
    renderer = DiagramRenderer()
    spec = _simple_spec(width=900, height=420)
    spec.nodes.extend(
        [
            DiagramNode(id="a", label="A", x=0.25, y=0.5),
            DiagramNode(id="b", label="B", x=0.75, y=0.5),
        ]
    )
    spec.edges.append(DiagramEdge(source="a", target="b", label="dashed", style="dashed"))
    data = renderer.render(spec)
    _assert_png_bytes(data)


def test_render_to_b64() -> None:
    renderer = DiagramRenderer()
    b64 = renderer.render_to_b64(_simple_spec())
    assert b64.startswith("data:image/png;base64,")
    decoded = base64.b64decode(b64.split(",", 1)[1])
    _assert_png_bytes(decoded)


def test_render_with_sublabels() -> None:
    renderer = DiagramRenderer()
    spec = _simple_spec(width=900, height=420)
    spec.nodes.append(
        DiagramNode(
            id="node",
            label="DiagramRenderer",
            sublabel="render(spec) -> bytes",
            x=0.5,
            y=0.5,
            width=0.6,
            height=0.25,
            shape="rounded_rect",
        )
    )
    data = renderer.render(spec)
    _assert_png_bytes(data)


def test_render_long_labels() -> None:
    renderer = DiagramRenderer()
    spec = _simple_spec(width=1100, height=520)
    spec.nodes.append(
        DiagramNode(
            id="long",
            label="DiagramRenderer renders DiagramSpec into PNG bytes and supports title, groups, shapes, and edges",
            x=0.5,
            y=0.5,
            width=0.8,
            height=0.35,
            shape="rounded_rect",
        )
    )
    data = renderer.render(spec)
    _assert_png_bytes(data)


def test_render_diamond_node() -> None:
    renderer = DiagramRenderer()
    spec = _simple_spec(width=800, height=420)
    spec.nodes.append(
        DiagramNode(
            id="decision",
            label="Pass?",
            x=0.5,
            y=0.5,
            width=0.35,
            height=0.28,
            shape="diamond",
            color="#fef3c7",
            border_color="#f59e0b",
        )
    )
    data = renderer.render(spec)
    _assert_png_bytes(data)


def test_render_architecture_diagram() -> None:
    renderer = DiagramRenderer()
    spec = DiagramSpec(
        title="Art Director MCP — Module Architecture",
        width=1400,
        height=900,
        background_color="#0b1020",
        font_family="DejaVu Sans",
    )

    spec.nodes.extend(
        [
            DiagramNode(
                id="server",
                label="server.py",
                sublabel="MCP tools/resources",
                x=0.5,
                y=0.12,
                group="mcp",
                color="#1d4ed8",
            ),
            DiagramNode(
                id="pipeline",
                label="pipeline.py",
                sublabel="orchestrator",
                x=0.5,
                y=0.28,
                group="core",
                color="#0ea5e9",
            ),
            DiagramNode(
                id="planner",
                label="planner.py",
                sublabel="LLM planning",
                x=0.2,
                y=0.48,
                group="agents",
                color="#22c55e",
            ),
            DiagramNode(
                id="executor",
                label="executor.py",
                sublabel="HF inference",
                x=0.5,
                y=0.48,
                group="agents",
                color="#f59e0b",
            ),
            DiagramNode(
                id="critic",
                label="critic.py",
                sublabel="CLIP + VLM audit",
                x=0.8,
                y=0.48,
                group="agents",
                color="#a855f7",
            ),
            DiagramNode(
                id="registry",
                label="registry.py",
                sublabel="catalog + presets",
                x=0.2,
                y=0.70,
                group="core",
                color="#38bdf8",
            ),
            DiagramNode(
                id="schemas",
                label="schemas.py",
                sublabel="Pydantic types",
                x=0.5,
                y=0.70,
                group="core",
                color="#38bdf8",
            ),
            DiagramNode(
                id="utils", label="utils.py", sublabel="png/json helpers", x=0.8, y=0.70, group="core", color="#38bdf8"
            ),
        ]
    )

    spec.edges.extend(
        [
            DiagramEdge(source="server", target="pipeline", label="calls"),
            DiagramEdge(source="pipeline", target="planner", label="create_plan"),
            DiagramEdge(source="pipeline", target="executor", label="execute"),
            DiagramEdge(source="pipeline", target="critic", label="audit"),
            DiagramEdge(source="planner", target="registry", label="model list", style="dashed"),
            DiagramEdge(source="executor", target="registry", label="fallback chain", style="dashed"),
            DiagramEdge(source="pipeline", target="schemas", label="types", style="dashed"),
        ]
    )

    spec.groups = {
        "mcp": {"label": "MCP", "color": "#1d4ed8"},
        "core": {"label": "Core", "color": "#0ea5e9"},
        "agents": {"label": "Agents", "color": "#22c55e"},
    }

    data = renderer.render(spec)
    _assert_png_bytes(data, min_size=10_000)


def test_render_custom_colors() -> None:
    renderer = DiagramRenderer()
    spec = DiagramSpec(width=900, height=500, background_color="#111827", font_family="DejaVu Sans")
    spec.nodes.extend(
        [
            DiagramNode(
                id="a",
                label="A",
                x=0.3,
                y=0.5,
                color="#f97316",
                text_color="#0b1020",
                border_color="#7c2d12",
                shape="ellipse",
            ),
            DiagramNode(
                id="b",
                label="B",
                x=0.7,
                y=0.5,
                color="#22c55e",
                text_color="#0b1020",
                border_color="#14532d",
                shape="hexagon",
            ),
        ]
    )
    spec.edges.append(DiagramEdge(source="a", target="b", label="custom", color="#e5e7eb", arrow=True))
    data = renderer.render(spec)
    _assert_png_bytes(data)


def test_render_resolution() -> None:
    renderer = DiagramRenderer()
    spec = DiagramSpec(width=1234, height=777)
    data = renderer.render(spec)
    size = _assert_png_bytes(data)
    assert size == (spec.width, spec.height)
