"""The published-weights helpers.

The real checkpoint is 348 MB and lives on a gated Hub repo, so the loading
path is exercised against a locally built state dict rather than a download.
`tests/test_methods_smoke.py` covers the network path under `-m slow`.
"""

from __future__ import annotations

import json

import pytest

from fbp_benchmark import load_pretrained, predict

torch = pytest.importorskip("torch")


@pytest.fixture
def checkpoint(tmp_path):
    """An untrained dinov2-partial, saved in the published layout."""
    from fbp_benchmark.registry import create

    method = create("dinov2-partial")
    method.build_modules()
    path = tmp_path / "dinov2-partial.pt"
    torch.save({k: m.state_dict() for k, m in method._modules().items()}, path)
    return path


@pytest.mark.slow
def test_load_pretrained_restores_every_module(checkpoint):
    model = load_pretrained("dinov2-partial", weights=checkpoint)
    assert set(model._modules()) == {"backbone", "head"}
    assert not model.head.training, "must be in eval mode"


@pytest.mark.slow
def test_predict_returns_a_score_in_range(checkpoint):
    from PIL import Image

    model = load_pretrained("dinov2-partial", weights=checkpoint)
    image = Image.new("RGB", (256, 256), (128, 128, 128))
    score = predict(model, image)
    assert isinstance(score, float)
    assert 1.0 <= score <= 10.0


@pytest.mark.slow
def test_predict_accepts_a_batch_and_returns_an_array(checkpoint):
    from PIL import Image

    model = load_pretrained("dinov2-partial", weights=checkpoint)
    images = [Image.new("RGB", (256, 256), (c, c, c)) for c in (0, 128, 255)]
    scores = predict(model, images)
    assert scores.shape == (3,)


def test_a_checkpoint_missing_a_module_is_rejected(tmp_path):
    bad = tmp_path / "dinov2-partial.pt"
    torch.save({"head": {}}, bad)  # no backbone
    with pytest.raises(ValueError, match="missing module"):
        load_pretrained("dinov2-partial", weights=bad)


def test_a_method_with_no_tensors_says_so(tmp_path):
    empty = tmp_path / "mean-baseline.pt"
    torch.save({}, empty)
    with pytest.raises(TypeError, match="no weights to load"):
        load_pretrained("mean-baseline", weights=empty)


def test_an_unpublished_method_names_what_is_published(tmp_path, monkeypatch):
    """The config lists what the repo holds, so a typo fails before the download."""
    from fbp_benchmark import pretrained

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"methods": {"dinov2-partial": {}}}))
    monkeypatch.setattr(
        pretrained, "_hub_file", lambda repo_id, filename, revision: str(config)
    )
    with pytest.raises(KeyError, match="dinov2-partial"):
        load_pretrained("cnn-resnet18")


def test_the_config_is_fetched_before_the_weights(tmp_path, monkeypatch):
    """Hub download counting keys on config.json, so it must be requested."""
    from fbp_benchmark import pretrained

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"methods": {}}))
    asked: list[str] = []

    def fake(repo_id, filename, revision):
        asked.append(filename)
        if filename != pretrained.CONFIG_FILE:
            raise RuntimeError("stop before the 348 MB download")
        return str(config)

    monkeypatch.setattr(pretrained, "_hub_file", fake)
    with pytest.raises(RuntimeError, match="stop before"):
        load_pretrained("dinov2-partial")
    assert asked[0] == "config.json", asked


def test_align_state_translates_both_dinov2_attention_layouts():
    """transformers 5.18 renamed these; checkpoints predate the rename."""
    from fbp_benchmark.pretrained import align_state

    class Fake:
        def __init__(self, keys):
            self._keys = keys

        def state_dict(self):
            return dict.fromkeys(self._keys, 0)

    new_style = [
        "model.encoder.layer.0.attention.q_proj.weight",
        "model.encoder.layer.0.attention.o_proj.bias",
    ]
    old_style = [
        "model.encoder.layer.0.attention.attention.query.weight",
        "model.encoder.layer.0.attention.output.dense.bias",
    ]
    old_checkpoint = dict.fromkeys(old_style, 1)
    assert set(align_state(Fake(new_style), old_checkpoint)) == set(new_style)

    new_checkpoint = dict.fromkeys(new_style, 1)
    assert set(align_state(Fake(old_style), new_checkpoint)) == set(old_style)

    # Already aligned, and unrelated keys, are returned untouched.
    assert align_state(Fake(new_style), new_checkpoint) is new_checkpoint
    other = {"backbone.fc.weight": 1}
    assert align_state(Fake(["backbone.fc.weight"]), other) is other
