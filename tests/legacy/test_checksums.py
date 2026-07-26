import hashlib
from pathlib import Path

from mebeauty_benchmark.legacy.checksums import group_files_by_sha256, sha256_file


def test_sha256_file_matches_hashlib(tmp_path: Path):
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"mebeauty" * 1000)

    expected = hashlib.sha256(file_path.read_bytes()).hexdigest()

    assert sha256_file(file_path) == expected


def test_group_files_by_sha256_finds_byte_identical_files_under_different_names(
    tmp_path: Path,
):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "original.jpg").write_bytes(b"same content")
    (tmp_path / "b" / "renamed-from-stock-site.jpg").write_bytes(b"same content")
    (tmp_path / "a" / "unique.jpg").write_bytes(b"different content")

    groups = group_files_by_sha256(tmp_path)
    duplicate_groups = [paths for paths in groups.values() if len(paths) > 1]

    assert len(duplicate_groups) == 1
    assert {p.name for p in duplicate_groups[0]} == {
        "original.jpg",
        "renamed-from-stock-site.jpg",
    }


def test_group_files_by_sha256_no_duplicates_when_all_unique(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"one")
    (tmp_path / "b.jpg").write_bytes(b"two")

    groups = group_files_by_sha256(tmp_path)

    assert all(len(paths) == 1 for paths in groups.values())
