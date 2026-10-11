import vaezip


def test_version_is_string() -> None:
    assert isinstance(vaezip.__version__, str)
