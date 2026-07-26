from mebeauty_benchmark.legacy.paths import normalize_image_path


def test_strips_ubuntu_crop_prefix():
    assert (
        normalize_image_path("/home/ubuntu/crop/male/black/x.jpg") == "male/black/x.jpg"
    )


def test_strips_me_beautydatabase_prefix():
    path = "/home/ubuntu/ME-beautydatabase/images/female/indian/x.jpg"
    assert normalize_image_path(path) == "female/indian/x.jpg"


def test_strips_mebeauty_database_prefix():
    path = "/home/ubuntu/MEBeauty-database/images/male/indian/x.csv"
    assert normalize_image_path(path) == "male/indian/x.csv"


def test_strips_relative_cropped_images_backend_dir():
    path = "./cropped_images/images_crop_align_mtcnn/female/indian/x.jpg"
    assert normalize_image_path(path) == "female/indian/x.jpg"


def test_strips_cropped_images_backend_dir_without_leading_dot():
    path = "cropped_images/images_crop_align_opencv/male/asian/x.jpg"
    assert normalize_image_path(path) == "male/asian/x.jpg"


def test_strips_leading_dot_slash():
    assert normalize_image_path("./male/asian/x.jpg") == "male/asian/x.jpg"


def test_leaves_already_relative_paths_unchanged():
    assert normalize_image_path("female/caucasian/x.jpg") == "female/caucasian/x.jpg"
