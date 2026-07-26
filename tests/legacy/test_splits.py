from mebeauty_benchmark.legacy.splits import (
    Rating,
    dedupe_across_splits,
    dedupe_by_content_hash,
    parse_split_lines,
    resolve_to_existing_images,
)


def test_parse_split_lines_reads_path_and_score():
    lines = ["female/caucasian/f8.jpg 7.663194444444445", "", "male/asian/x.jpg 4.5"]
    ratings = parse_split_lines(lines)
    assert ratings == [
        Rating(path="female/caucasian/f8.jpg", score=7.663194444444445),
        Rating(path="male/asian/x.jpg", score=4.5),
    ]


def test_parse_split_lines_strips_quotes_around_paths_with_spaces():
    lines = ['"female/mideastern/kamal-alkhatib (1).jpg" 7.444444444444445']
    ratings = parse_split_lines(lines)
    assert ratings == [
        Rating(path="female/mideastern/kamal-alkhatib (1).jpg", score=7.444444444444445)
    ]


def test_resolve_to_existing_images_keeps_valid_paths():
    splits = {"train": [Rating(path="a/x.jpg", score=1.0)]}
    resolved, issues = resolve_to_existing_images(
        splits, valid_paths={"a/x.jpg"}, basename_to_paths={}
    )
    assert [r.path for r in resolved["train"]] == ["a/x.jpg"]
    assert issues == []


def test_resolve_to_existing_images_remaps_relabeled_image():
    # Rated as male/caucasian/x.jpg, but the file now lives under male/asian/.
    splits = {"train": [Rating(path="male/caucasian/x.jpg", score=1.0)]}
    resolved, issues = resolve_to_existing_images(
        splits,
        valid_paths={"male/asian/x.jpg"},
        basename_to_paths={"x.jpg": ["male/asian/x.jpg"]},
    )
    assert [r.path for r in resolved["train"]] == ["male/asian/x.jpg"]
    assert issues == [
        {
            "split": "train",
            "rated_path": "male/caucasian/x.jpg",
            "status": "relabeled",
            "resolved_path": "male/asian/x.jpg",
        }
    ]


def test_resolve_to_existing_images_drops_image_that_no_longer_exists_anywhere():
    splits = {"train": [Rating(path="male/caucasian/gone.jpg", score=1.0)]}
    resolved, issues = resolve_to_existing_images(
        splits, valid_paths=set(), basename_to_paths={}
    )
    assert resolved["train"] == []
    assert issues == [
        {
            "split": "train",
            "rated_path": "male/caucasian/gone.jpg",
            "status": "not_found",
        }
    ]


def test_resolve_to_existing_images_recovers_truncated_filename():
    # Rated under the full name; the file is stored with its stem cut to 32
    # characters ("...-unsplash.jpg" -> "...-unspla.jpg").
    rated = "male/black/payton-tuttle-n_RdRxH_7h4-unsplash.jpg"
    stored = "male/black/payton-tuttle-n_RdRxH_7h4-unspla.jpg"
    resolved, issues = resolve_to_existing_images(
        {"train": [Rating(path=rated, score=3.5)]},
        valid_paths={stored},
        basename_to_paths={"payton-tuttle-n_rdrxh_7h4-unspla.jpg": [stored]},
    )
    assert [r.path for r in resolved["train"]] == [stored]
    assert issues == [
        {
            "split": "train",
            "rated_path": rated,
            "status": "truncated_filename",
            "resolved_path": stored,
        }
    ]


def test_truncation_recovery_declines_when_full_named_file_also_exists():
    # Real case: jonathan-borba-5rQG1mib90I-unspl.jpg and -unsplash.jpg are two
    # DIFFERENT photos. The rating belongs to the full-named one, so the
    # truncated file must not inherit it.
    rated = "female/mideastern/jonathan-borba-5rQG1mib90I-unsplash.jpg"
    truncated = "female/mideastern/jonathan-borba-5rQG1mib90I-unspl.jpg"
    resolved, issues = resolve_to_existing_images(
        {"train": [Rating(path=rated, score=7.6)]},
        valid_paths={rated, truncated},
        basename_to_paths={
            "jonathan-borba-5rqg1mib90i-unsplash.jpg": [rated],
            "jonathan-borba-5rqg1mib90i-unspl.jpg": [truncated],
        },
    )
    # Resolves to the full-named file it actually names, not the truncated one.
    assert [r.path for r in resolved["train"]] == [rated]
    assert issues == []


def test_truncation_recovery_declines_when_candidate_is_ambiguous():
    rated = "male/black/some-very-long-filename-here-unsplash.jpg"
    resolved, issues = resolve_to_existing_images(
        {"train": [Rating(path=rated, score=1.0)]},
        valid_paths={"a/some-very-long-filename-here-uns.jpg", "b/x.jpg"},
        basename_to_paths={
            "some-very-long-filename-here-uns.jpg": [
                "a/some-very-long-filename-here-uns.jpg",
                "b/some-very-long-filename-here-uns.jpg",
            ]
        },
    )
    assert resolved["train"] == []
    assert issues[0]["status"] == "not_found"


def test_resolve_to_existing_images_drops_ambiguous_basename_rather_than_guessing():
    splits = {"train": [Rating(path="male/caucasian/dup.jpg", score=1.0)]}
    resolved, issues = resolve_to_existing_images(
        splits,
        valid_paths=set(),
        basename_to_paths={"dup.jpg": ["female/asian/dup.jpg", "male/black/dup.jpg"]},
    )
    assert resolved["train"] == []
    assert issues == [
        {
            "split": "train",
            "rated_path": "male/caucasian/dup.jpg",
            "status": "ambiguous",
            "candidates": "female/asian/dup.jpg, male/black/dup.jpg",
        }
    ]


def test_dedupe_across_splits_keeps_first_by_priority():
    splits = {
        "train": [
            Rating(path="a/dup.jpg", score=5.0),
            Rating(path="a/only_train.jpg", score=1.0),
        ],
        "val": [Rating(path="a/dup.jpg", score=6.0)],
        "test": [Rating(path="c/only_test.jpg", score=2.0)],
    }

    deduped, removed = dedupe_across_splits(splits, priority=["train", "val", "test"])

    assert [r.path for r in deduped["train"]] == ["a/dup.jpg", "a/only_train.jpg"]
    assert deduped["val"] == []
    assert [r.path for r in deduped["test"]] == ["c/only_test.jpg"]
    assert removed == [("a/dup.jpg", "val")]


def test_dedupe_across_splits_is_noop_when_no_overlap():
    splits = {
        "train": [Rating(path="a.jpg", score=1.0)],
        "val": [Rating(path="b.jpg", score=2.0)],
    }
    deduped, removed = dedupe_across_splits(splits, priority=["train", "val"])
    assert deduped == splits
    assert removed == []


def test_dedupe_across_splits_also_drops_within_split_repeats():
    # Same full path appears three times in train and once in val.
    splits = {
        "train": [
            Rating(path="a/dup.jpg", score=1.0),
            Rating(path="a/dup.jpg", score=1.5),
            Rating(path="a/dup.jpg", score=1.5),
        ],
        "val": [Rating(path="a/dup.jpg", score=2.0)],
    }

    deduped, removed = dedupe_across_splits(splits, priority=["train", "val"])

    assert [r.path for r in deduped["train"]] == ["a/dup.jpg"]
    assert deduped["val"] == []
    assert removed == [
        ("a/dup.jpg", "train"),
        ("a/dup.jpg", "train"),
        ("a/dup.jpg", "val"),
    ]


def test_dedupe_across_splits_does_not_conflate_same_basename_in_different_folders():
    # Two genuinely different photos can share a filename across folders
    # (confirmed for one real image in this dataset). A basename-only key
    # would wrongly treat these as duplicates and drop one; the full
    # normalized path must keep both.
    splits = {
        "train": [Rating(path="male/indian/shivam-singh.jpg", score=5.89)],
        "test": [Rating(path="male/mideastern/shivam-singh.jpg", score=5.89)],
    }

    deduped, removed = dedupe_across_splits(splits, priority=["train", "test"])

    assert [r.path for r in deduped["train"]] == ["male/indian/shivam-singh.jpg"]
    assert [r.path for r in deduped["test"]] == ["male/mideastern/shivam-singh.jpg"]
    assert removed == []


def test_dedupe_across_splits_normalizes_paths_before_comparing():
    # The same image referenced via two different legacy path conventions
    # must still be recognized as one duplicate.
    splits = {
        "train": [Rating(path="/home/ubuntu/crop/male/black/x.jpg", score=1.0)],
        "val": [
            Rating(
                path="./cropped_images/images_crop_align_mtcnn/male/black/x.jpg",
                score=2.0,
            )
        ],
    }

    deduped, removed = dedupe_across_splits(splits, priority=["train", "val"])

    assert [r.path for r in deduped["train"]] == ["/home/ubuntu/crop/male/black/x.jpg"]
    assert deduped["val"] == []
    assert removed == [("male/black/x.jpg", "val")]


def test_dedupe_by_content_hash_catches_same_image_under_different_filenames():
    # basename-based dedup would miss this: different paths, same content hash.
    splits = {
        "train": [Rating(path="a/stock-photo-slug.jpg", score=5.0)],
        "test": [Rating(path="b/12.jpg", score=8.0)],
    }
    path_to_hash = {"a/stock-photo-slug.jpg": "hash1", "b/12.jpg": "hash1"}

    deduped, removed = dedupe_by_content_hash(
        splits, priority=["train", "test"], path_to_hash=path_to_hash
    )

    assert [r.path for r in deduped["train"]] == ["a/stock-photo-slug.jpg"]
    assert deduped["test"] == []
    assert removed == [("b/12.jpg", "hash1", "test")]


def test_dedupe_by_content_hash_leaves_unhashed_paths_alone():
    splits = {
        "train": [Rating(path="a.jpg", score=1.0)],
        "test": [Rating(path="b.jpg", score=2.0)],
    }

    deduped, removed = dedupe_by_content_hash(
        splits, priority=["train", "test"], path_to_hash={}
    )

    assert deduped == splits
    assert removed == []
