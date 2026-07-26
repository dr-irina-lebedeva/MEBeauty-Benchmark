from pathlib import Path

from mebeauty_benchmark.legacy.labels import find_cross_label_collisions


def test_finds_files_shared_across_label_folders(tmp_path: Path):
    (tmp_path / "male" / "indian").mkdir(parents=True)
    (tmp_path / "male" / "mideastern").mkdir(parents=True)
    (tmp_path / "male" / "indian" / "shared.jpg").write_bytes(b"a")
    (tmp_path / "male" / "mideastern" / "shared.jpg").write_bytes(b"a")
    (tmp_path / "male" / "indian" / "unique.jpg").write_bytes(b"b")

    conflicts = find_cross_label_collisions(tmp_path)

    assert len(conflicts) == 1
    assert conflicts[0].filename == "shared.jpg"
    assert len(conflicts[0].label_paths) == 2


def test_no_conflicts_when_all_filenames_unique(tmp_path: Path):
    (tmp_path / "female" / "asian").mkdir(parents=True)
    (tmp_path / "female" / "asian" / "a.jpg").write_bytes(b"a")
    (tmp_path / "female" / "asian" / "b.jpg").write_bytes(b"b")

    assert find_cross_label_collisions(tmp_path) == []
