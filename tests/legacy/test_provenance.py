from mebeauty_benchmark.legacy.provenance import infer_provenance


def test_infers_unsplash_from_suffix():
    result = infer_provenance("hossein-rezaei--paUf05gaUs-unsplash.jpg")
    assert result.platform == "unsplash"
    assert result.confidence == "inferred"
    assert result.inferred_source_url.startswith("https://unsplash.com/photos/")


def test_unsplash_id_not_truncated_when_id_contains_hyphen():
    # Unsplash ids are a fixed 11 chars and may contain "-" themselves;
    # splitting on the last hyphen would wrongly truncate these.
    result = infer_provenance("shivam-singh-2_X6NMP-E_U-unsplash.jpg")
    assert result.photo_id == "2_X6NMP-E_U"
    assert result.inferred_source_url == "https://unsplash.com/photos/2_X6NMP-E_U"

    result = infer_provenance("hossein-rezaei--paUf05gaUs-unsplash.jpg")
    assert result.photo_id == "-paUf05gaUs"


def test_infers_pexels_from_prefix_and_numeric_id():
    result = infer_provenance("pexels-cottonbro-5529905.jpg")
    assert result.platform == "pexels"
    assert result.photo_id == "5529905"
    assert result.inferred_source_url == "https://www.pexels.com/photo/5529905/"


def test_infers_pixabay_from_trailing_underscore_id():
    result = infer_provenance("woman-761642_1920.jpg")
    assert result.platform == "pixabay"
    assert result.photo_id == "761642"


def test_unknown_platform_for_bare_filenames():
    result = infer_provenance("f8.jpg")
    assert result.platform == "unknown"
    assert result.confidence == "unknown"
    assert result.inferred_source_url is None


def test_copy_marker_suffix_does_not_defeat_inference():
    # "(1)" / " (2)" are download-manager copy markers, not part of the
    # platform filename; leaving them in tagged obvious stock photos "unknown".
    for name in [
        "gift-habeshaw-KBv5dEN3QtY-unsplash(1).jpg",
        "kamal-alkhatib-IETO_Z0BrsE-unsplash (1).jpg",
    ]:
        result = infer_provenance(name)
        assert result.platform == "unsplash", name
        assert result.confidence == "inferred"

    assert infer_provenance("pexels-ethan-jones-3222422 (1).jpg").platform == "pexels"
    assert infer_provenance("outdoors-3462520_1920(1).jpg").platform == "pixabay"


def test_copy_marker_does_not_change_the_recovered_photo_id():
    assert (
        infer_provenance("gift-habeshaw-KBv5dEN3QtY-unsplash(1).jpg").photo_id
        == infer_provenance("gift-habeshaw-KBv5dEN3QtY-unsplash.jpg").photo_id
    )
