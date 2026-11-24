import colorsys

from evmlops.ui import palette


def _hls(hex_color: str):
    body = hex_color.lstrip("#")
    r, g, b = (int(body[i : i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, l, s


def test_brand_blue_becomes_prussian_blue():
    hue, lightness, saturation = _hls(palette.remap_hex("#2272b4"))
    assert 200 <= hue <= 210
    assert lightness < 0.22 and saturation > 0.7


def test_non_blue_colours_are_unchanged():
    for colour in ("#161616", "#e8ecf0", "#c82d4c", "#34824f", "#ffcb1f", "#077a9d", "#af5b23"):
        assert palette.remap_hex(colour) == colour


def test_light_tints_and_dark_theme_blues_keep_their_lightness():
    for colour in ("#d7ecfb", "#4299e0"):
        assert abs(_hls(palette.remap_hex(colour))[1] - _hls(colour)[1]) < 0.01


def test_alpha_and_case_are_preserved():
    assert palette.remap_hex("#2272b433").endswith("33")
    assert palette.remap_hex("#2272B4") == palette.remap_hex("#2272b4").upper()


def test_remap_text_handles_hex_and_rgb():
    out = palette.remap_text("a{color:#2272b4;b:rgba(34,114,180,.5);c:#c82d4c}")
    assert "#2272b4" not in out
    assert "34,114,180" not in out
    assert "#c82d4c" in out


def test_remap_text_ignores_invalid_rgb_channels():
    text = "rgb(999, 0, 0)"
    assert palette.remap_text(text) == text


def test_build_themed_static_is_complete_and_cached(tmp_path):
    source = tmp_path / "build"
    (source / "static" / "css").mkdir(parents=True)
    (source / "static" / "js").mkdir(parents=True)
    (source / "index.html").write_text(
        '<html><head><meta name="theme-color" content="#000000"/></head><body></body></html>'
    )
    (source / "static" / "css" / "main.css").write_text(".a{color:#2272b4}.b{color:#c82d4c}")
    (source / "static" / "js" / "main.js").write_text('const primary="#2272B4";')
    (source / "favicon.ico").write_bytes(b"\x00\x01")

    cache = tmp_path / "cache"
    built = palette.build_themed_static(source, "0.0.0", cache)

    html = (built / "index.html").read_text()
    assert "static-files/prussian-gold.css" in html
    assert 'content="#003153"' in html
    assert (built / "prussian-gold.css").is_file()
    assert "#2272b4" not in (built / "static" / "css" / "main.css").read_text()
    assert "#c82d4c" in (built / "static" / "css" / "main.css").read_text()
    assert "#2272B4" not in (built / "static" / "js" / "main.js").read_text()
    assert (built / "favicon.ico").read_bytes() == b"\x00\x01"

    again = palette.build_themed_static(source, "0.0.0", cache)
    assert again == built
    assert len(list(cache.iterdir())) == 1  # no stray staging directories
