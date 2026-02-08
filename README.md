# Art Director MCP

An MCP server that orchestrates a high-fidelity image generation pipeline with AI-powered planning, execution, and quality auditing.

## Overview

Art Director is an intelligent image generation orchestrator that combines multiple AI agents to produce high-quality images with minimal manual intervention. A Kimi k2.5 planner (via NVIDIA NIM) selects the optimal generation model and optimizes prompts, a Hugging Face executor runs the generation with intelligent fallback chains, and a hybrid CLIP+VLM critic audits results and feeds corrections back to the planner. The system automatically retries with prompt rewrites, model switches, or seed variations until quality targets are met or budgets are exhausted.

## Architecture

```
User Prompt
    |
    v
+-------------------+
| Planner (LLM)     |  Classifies intent, selects model,
| (Kimi k2.5 / NIM) |  optimizes prompt, configures params
+-------------------+
    |
    v
+-------------------+
| Executor (HF)     |  Waterfall: dedicated → serverless → fallback
| (FLUX, SDXL, SD3) |  Retries with exponential backoff
+-------------------+
    |
    v
+-------------------+
| Critic (CLIP+VLM) |  Fast-path CLIP scoring (0.1s)
| (Kimi k2.5 / NIM) |  Deep audit with VLM (2-8s)
+-------------------+
    |
    v
  Pass? -----> Return Image
    |
    No
    |
    v
  Refine Plan (prompt rewrite > model switch > seed change > param nudge)
    |
    v
  Retry (up to 3 attempts, budget permitting)
```

## Features

- **13 MCP Tools**: Full pipeline control, model comparison, batch generation, job management
- **3 MCP Resources**: Model catalog, style presets, dynamic metadata
- **7 Image Generation Models**: FLUX.1 (dev/schnell), SDXL, SD3.5 (large/turbo), Kolors, Playground v2.5
- **12 Style Presets**: photorealistic, cinematic, anime, technical-diagram, watercolor, oil-painting, pixel-art, logo-design, concept-art, minimalist, 3d-render, sketch
- **Intelligent Model Selection**: LLM-driven capability matching (text rendering, photorealism, artistic style, speed, consistency, 3D)
- **Hybrid Quality Auditing**: CLIP fast-path (0.1s) for quick filtering, VLM deep audit (2-8s) for nuanced evaluation
- **Feedback Loop**: Prompt rewriting, model switching, seed variation, parameter tuning with clear priority hierarchy
- **Predictive Budget Enforcement**: Cost and wall-clock checks before each attempt; job eviction at 1000 cap
- **A/B Model Comparison**: Generate same prompt with multiple models, compare scores, get recommendations
- **Batch Generation**: Parallel image generation with configurable count and budget
- **Async Job Management**: Non-blocking generation with progress tracking, cancellation support, job history
- **Extended Thinking**: Planner and Critic agents use extended thinking for deeper reasoning on model selection and quality assessment (requires supported LLM)

## Extended Thinking

Art Director enables **extended thinking** for the planner and critic LLM agents, allowing them to reason more deeply before making decisions.

### How It Works

- **Planner Agent**: When creating or refining a generation plan, the LLM uses extended thinking to:
  - Analyze the user's intent and constraints
  - Compare available models against capability requirements
  - Reason through the best prompt optimizations
  - Decide on parameter configurations

- **Critic Agent**: When auditing generated images, the LLM uses extended thinking to:
  - Carefully analyze image content against the prompt
  - Reason through quality issues and their severity
  - Provide actionable feedback for refinement

### Configuration

Extended thinking is **disabled by default** in both agents. To enable it, set the environment variable to `true`:

```python
response = await client.chat.completions.create(
    model=settings.planner_model,
    messages=[...],
    temperature=0.3,
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"thinking": False}},  # Extended thinking disabled by default
)
```

To enable extended thinking, set `ART_DIRECTOR_PLANNER_THINKING_ENABLED=true` or `ART_DIRECTOR_CRITIC_THINKING_ENABLED=true` in your `.env` file.

### Supported Models

Extended thinking works best with models that support the `thinking` parameter:
- **NVIDIA NIM**: Kimi k2.5 and other reasoning-capable models
- **OpenAI**: o1, o3 (with appropriate API versions)
- **Other OpenAI-compatible endpoints**: Check your provider's documentation

### Performance Impact

Extended thinking increases latency:
- Planner planning: 2-5s → 5-15s (deeper reasoning)
- Critic auditing: 2-8s → 5-20s (more thorough analysis)

The wall-clock budget (`ART_DIRECTOR_MAX_WALL_CLOCK_SECONDS`, default 300s) accounts for this overhead.

### Disabling Extended Thinking

If your LLM doesn't support extended thinking or you want faster responses, edit `src/art_director/planner.py` and `src/art_director/critic.py` to remove the `extra_body` parameter from `chat.completions.create()` calls.

## Quick Start

### Installation

```bash
git clone https://github.com/slapglif/art-director-mcp.git
cd art-director-mcp
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env and fill in your API keys:
# - PLANNER_API_KEY (NVIDIA NIM key for Kimi k2.5, or any OpenAI-compatible key)
# - HF_API_TOKEN (Hugging Face)
# - CRITIC_API_KEY (NVIDIA NIM key for Kimi k2.5)
```

### Run

```bash
python -m art_director
# Server starts on http://localhost:8000
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ART_DIRECTOR_PLANNER_API_KEY` | (required) | API key for planner LLM (OpenAI-compatible) |
| `ART_DIRECTOR_PLANNER_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Planner endpoint (NVIDIA NIM default, OpenAI-compatible) |
| `ART_DIRECTOR_PLANNER_MODEL` | `moonshotai/kimi-k2.5` | Planner model (Kimi k2.5 on NVIDIA NIM) |
| `ART_DIRECTOR_PLANNER_THINKING_ENABLED` | `false` | Enable extended thinking for planner (requires supported model) |
| `ART_DIRECTOR_HF_API_TOKEN` | (required) | Hugging Face API token |
| `ART_DIRECTOR_CRITIC_API_KEY` | (required) | API key for critic VLM (NVIDIA NIM for Kimi k2.5) |
| `ART_DIRECTOR_CRITIC_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Critic endpoint (NVIDIA NIM) |
| `ART_DIRECTOR_CRITIC_MODEL` | `moonshotai/kimi-k2.5` | Critic model (Kimi k2.5 on NVIDIA NIM) |
| `ART_DIRECTOR_CRITIC_THINKING_ENABLED` | `false` | Enable extended thinking for critic (requires supported model) |
| `ART_DIRECTOR_NIM_API_KEY` | (optional) | Shared NVIDIA NIM API key (fallback for planner + critic if their keys are unset) |
| `ART_DIRECTOR_CLIP_ENABLED` | `false` | Enable CLIP fast-path scoring |
| `ART_DIRECTOR_CLIP_THRESHOLD_PASS` | `0.82` | CLIP score for auto-pass |
| `ART_DIRECTOR_CLIP_THRESHOLD_FAIL` | `0.55` | CLIP score for auto-fail |
| `ART_DIRECTOR_MAX_RETRIES` | `3` | Maximum generation attempts |
| `ART_DIRECTOR_MAX_WALL_CLOCK_SECONDS` | `300` | Wall-clock timeout (5 min) |
| `ART_DIRECTOR_MAX_COST_USD` | `1.00` | Maximum spend per request |
| `ART_DIRECTOR_DEFAULT_WIDTH` | `1024` | Default image width |
| `ART_DIRECTOR_DEFAULT_HEIGHT` | `1024` | Default image height |
| `ART_DIRECTOR_TRANSPORT` | `streamable-http` | MCP transport (stdio, streamable-http, sse) |
| `ART_DIRECTOR_HOST` | `0.0.0.0` | Server bind address |
| `ART_DIRECTOR_PORT` | `8000` | Server port |
| `ART_DIRECTOR_OUTPUT_DIR` | `./generated` | Image output directory |

## MCP Tools

| Tool | Description |
|------|-------------|
| `generate` | Full pipeline: plan → execute → audit → retry. Returns image, scores, and metadata. |
| `generate_with_reference` | Generate with reference image for style/composition guidance (img2img). |
| `draft_plan` | Preview planner decision without executing (model selection, prompt optimization). |
| `audit_image` | Audit an existing image against a prompt (CLIP + VLM scoring). |
| `compare_models` | Generate same prompt with multiple models, compare quality scores. |
| `batch_generate` | Generate N variations of a prompt in parallel. |
| `estimate_cost` | Estimate cost before running (single attempt + full pipeline with retries). |
| `list_models` | List all available image generation models with capabilities. |
| `list_style_presets` | List all available style presets. |
| `check_model_health` | Check if model(s) are available and responding. |
| `get_job_status` | Poll status of a running generation job. |
| `cancel_job` | Cancel a running job. |
| `list_jobs` | List recent jobs, optionally filtered by state. |

## MCP Resources

| Resource | Description |
|----------|-------------|
| `model://catalog` | Full model catalog with capabilities, availability, cost, and metadata. |
| `model://catalog/{model_id}` | Detailed info for a specific model (URL-encoded model ID). |
| `style://presets` | All available style presets with prompt suffixes and negative prompts. |

## Style Presets

- **photorealistic**: 8k uhd, high detail, sharp focus, DSLR quality
- **cinematic**: Cinematic lighting, dramatic shadows, film grain, anamorphic lens
- **anime**: Cel shading, vibrant colors, detailed linework, studio quality
- **technical-diagram**: Clean lines, labeled components, white background, vector style
- **watercolor**: Soft edges, flowing pigments, paper texture, artistic brushstrokes
- **oil-painting**: Thick brushstrokes, rich colors, canvas texture, classical style
- **pixel-art**: 16-bit style, retro gaming aesthetic, clean pixels, limited palette
- **logo-design**: Minimalist, scalable, clean vector, professional branding
- **concept-art**: Digital painting, matte painting, detailed environment, epic composition
- **minimalist**: Simple shapes, limited color palette, clean composition, negative space
- **3d-render**: Octane render, volumetric lighting, subsurface scattering, ray tracing
- **sketch**: Pencil sketch, hand-drawn, charcoal, line art, crosshatching

## How It Works

1. **User submits prompt** with optional style, dimensions, and budget constraints
2. **Planner (LLM)** classifies intent, selects best model from registry, optimizes prompt, configures parameters
3. **Executor** runs generation via HF Inference with waterfall fallback (dedicated → serverless → fallback model)
4. **Critic** scores image with CLIP fast-path (0.1s); if ambiguous, runs VLM deep audit (2-8s)
5. **Quality check**: If score ≥ 7.0 → PASS, return image; if ≤ 4.0 → FAIL, refine
6. **Refinement** (if needed): Planner rewrites prompt, switches model, or changes seed based on feedback
7. **Retry loop**: Repeat steps 3-6 up to max_retries, respecting cost and wall-clock budgets
8. **Return**: Best image found, audit scores, attempt count, total cost, and feedback

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Testing

```bash
pytest tests/
pytest --cov=src/art_director tests/
```

### Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Project Structure

```
art-director-mcp/
├── src/art_director/
│   ├── __main__.py          # Entry point
│   ├── config.py            # Pydantic Settings (env-driven)
│   ├── schemas.py           # Data models (20+ Pydantic classes)
│   ├── utils.py             # Helpers (image, JSON, cost, retry)
│   ├── registry.py          # Model catalog, health checks, waterfall
│   ├── planner.py           # LLM-based planning agent
│   ├── executor.py          # HF Inference executor with fallback
│   ├── critic.py            # CLIP + VLM auditing agent
│   ├── pipeline.py          # Orchestrator (retry loop, budgets, A/B, batch)
│   └── server.py            # MCP server (13 tools, 3 resources)
├── tests/
│   ├── test_schemas.py      # Schema validation tests
│   └── test_utils.py        # Utility function tests
├── pyproject.toml           # Project metadata and dependencies
├── .env.example             # Configuration template
└── README.md                # This file
```

## License

MIT
