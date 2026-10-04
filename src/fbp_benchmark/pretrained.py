"""Load a published checkpoint from the Hub and score an image with it.

    from fbp_benchmark import load_pretrained, predict

    model = load_pretrained("dinov2-partial")
    score = predict(model, "face.jpg")

The preprocessing here is the same as the harness uses at evaluation time --
resize to 256, centre crop, ImageNet statistics, test-time horizontal flip
averaging, clipped to the dataset's score range. A snippet that skips the flip
or resizes straight to the input size produces different numbers from the
published metrics, which is the mistake this module exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .registry import create

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

#: Where published weights live.
DEFAULT_MODEL_REPO = "dr-irina-lebedeva/MEBeauty-FBP-models"

#: Downloaded alongside the weights. The Hub counts a download per request to
#: `config.json` for repositories with no library-specific rule, so fetching
#: it is both how a caller learns the architecture and how the download is
#: counted. See https://huggingface.co/docs/hub/models-download-stats
CONFIG_FILE = "config.json"


def _hub_file(repo_id: str, filename: str, revision: str | None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, revision=revision)


#: transformers 5.18 renamed DINOv2's attention submodules. A checkpoint saved
#: before the rename cannot be loaded after it, and vice versa, so the two
#: layouts are translated here rather than pinning every user to one version.
#: Left is pre-5.18, right is 5.18+.
_ATTENTION_RENAMES: tuple[tuple[str, str], ...] = (
    ("attention.attention.query.", "attention.q_proj."),
    ("attention.attention.key.", "attention.k_proj."),
    ("attention.attention.value.", "attention.v_proj."),
    ("attention.output.dense.", "attention.o_proj."),
)


def _rename(state: dict[str, Any], forward: bool) -> dict[str, Any]:
    pairs = (
        _ATTENTION_RENAMES
        if forward
        else tuple((new, old) for old, new in _ATTENTION_RENAMES)
    )
    renamed = {}
    for key, value in state.items():
        for src, dst in pairs:
            if src in key:
                key = key.replace(src, dst)
                break
        renamed[key] = value
    return renamed


def align_state(module: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Translate a module's state dict between DINOv2 attention layouts.

    Returned unchanged when it already matches, so a method whose backbone is
    not a transformers model is unaffected.
    """
    expected = set(module.state_dict())
    if not expected - set(state):
        return state
    for forward in (True, False):
        candidate = _rename(state, forward)
        if not expected - set(candidate):
            return candidate
    return state


def load_pretrained(
    name: str = "dinov2-partial",
    repo_id: str = DEFAULT_MODEL_REPO,
    revision: str | None = None,
    weights: str | Path | None = None,
) -> Any:
    """A registered method with published weights loaded into it.

    `name` is the method's benchmark name. `weights` loads a local checkpoint
    instead of downloading one, which is what the tests use; otherwise
    `<name>.pt` is fetched from `repo_id`.

    The returned object is the method itself, in eval mode. Pass it to
    `predict`.
    """
    import torch

    config: dict[str, Any] = {}
    if weights is None:
        # Fetched before the weights so the download is counted even if the
        # caller interrupts the (much larger) checkpoint transfer.
        config = json.loads(
            Path(_hub_file(repo_id, CONFIG_FILE, revision)).read_text(encoding="utf-8")
        )
        published = config.get("methods", {})
        if published and name not in published:
            raise KeyError(
                f"{repo_id} publishes {sorted(published)}, not {name!r}. "
                "Train it locally with "
                f"`fbp-benchmark run --method {name} --save-weights checkpoints`."
            )
        weights = _hub_file(repo_id, f"{name}.pt", revision)

    # `create` is typed as the Method protocol, which deliberately knows
    # nothing about torch modules; the trainable methods carry them.
    method: Any = create(name)
    if not hasattr(method, "build_modules"):
        raise TypeError(
            f"{name!r} has no weights to load: the classical methods and the "
            "baseline hold no tensors."
        )
    method.build_modules()

    state = torch.load(weights, map_location=method.device, weights_only=True)
    modules = method._modules()
    missing = set(modules) - set(state)
    if missing:
        raise ValueError(
            f"Checkpoint for {name!r} is missing module(s) {sorted(missing)}; "
            f"it has {sorted(state)}. Wrong method name, or a stale file."
        )
    for key, module in modules.items():
        module.load_state_dict(align_state(module, state[key]))
        module.eval()

    method.pretrained_config = config
    return method


def _as_images(images: Any) -> tuple[list[Image], bool]:
    """Normalise one-or-many paths/PIL images to a list, plus a was-single flag."""
    from PIL import Image as PILImage

    single = not isinstance(images, (list, tuple))
    batch = [images] if single else list(images)
    loaded = [
        item.convert("RGB")
        if hasattr(item, "convert")
        else PILImage.open(item).convert("RGB")
        for item in batch
    ]
    return loaded, single


def predict(
    model: Any,
    images: Any,
    flip_average: bool = True,
) -> Any:
    """Score one image, or a list of them, on the dataset's 1-10 scale.

    Accepts paths or PIL images. Returns a float for a single input and a
    numpy array for a list. `flip_average=False` drops the test-time flip,
    which is faster and does not match the published metrics.
    """
    import torch
    from torchvision import transforms

    from .methods.training import MEAN, STD

    prepared, single = _as_images(images)
    size = model.config.image_size
    preprocess = transforms.Compose(
        [
            # Resize then centre crop, matching FaceDataset at eval time.
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    pixels = torch.stack([preprocess(image) for image in prepared]).to(model.device)

    with torch.no_grad():
        scores = model.to_scores(model._forward(pixels))
        if flip_average:
            flipped = model.to_scores(model._forward(torch.flip(pixels, dims=[3])))
            scores = 0.5 * (scores + flipped)

    # The same range the harness clips to; MEBeauty's scale is 1-10.
    config = getattr(model, "pretrained_config", None) or {}
    low, high = config.get("score_range", (1.0, 10.0))
    values = np.clip(scores.detach().cpu().numpy(), low, high)
    return float(values[0]) if single else values
