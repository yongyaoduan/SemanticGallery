from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "src-tauri" / "icons"
ICONSET_LAYOUT = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512,
    "icon_512x512@2x.png": 1024,
}
EXPORT_LAYOUT = {
    "icon.png": 1024,
    "128x128.png": 128,
    "128x128@2x.png": 256,
    "32x32.png": 32,
}


def _hex(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    return (*ImageColor.getrgb(color), alpha)


def _rounded_rect_mask(size: int, margin: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((margin, margin, size - margin, size - margin), radius=radius, fill=255)
    return mask


def _linear_gradient(width: int, height: int, top_left: str, bottom_right: str) -> Image.Image:
    start = _hex(top_left)
    end = _hex(bottom_right)
    gradient = Image.new("RGBA", (width, height))
    pixels = []
    for y in range(height):
        for x in range(width):
            ratio = (x / max(1, width - 1) + y / max(1, height - 1)) / 2
            pixels.append(
                tuple(
                    round(start[index] + (end[index] - start[index]) * ratio)
                    for index in range(4)
                )
            )
    gradient.putdata(pixels)
    return gradient


def _ellipse_overlay(size: int, bounds: tuple[int, int, int, int], color: str, alpha: int, blur: float) -> Image.Image:
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse(bounds, fill=_hex(color, alpha))
    return overlay.filter(ImageFilter.GaussianBlur(blur))


def _draw_photo_card(canvas: Image.Image, scale: float) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    back_cards = [
        ((228, 216, 612, 632), 74, "#f4e5c8", "#c7b69a", 16),
        ((272, 188, 660, 604), 78, "#f9efe0", "#d1c0a5", 10),
    ]
    for bounds, radius, fill, outline, rotation in back_cards:
        card = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        card_draw = ImageDraw.Draw(card, "RGBA")
        scaled = tuple(round(value * scale) for value in bounds)
        card_draw.rounded_rectangle(
            scaled,
            radius=round(radius * scale),
            fill=_hex(fill),
            outline=_hex(outline, 170),
            width=max(2, round(4 * scale)),
        )
        rotated = card.rotate(rotation, resample=Image.Resampling.BICUBIC)
        canvas.alpha_composite(rotated)

    card = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    card_draw = ImageDraw.Draw(card, "RGBA")
    main_bounds = tuple(round(value * scale) for value in (252, 230, 716, 760))
    radius = round(88 * scale)
    card_draw.rounded_rectangle(
        main_bounds,
        radius=radius,
        fill=_hex("#fff9ef"),
        outline=_hex("#cbb99d", 200),
        width=max(3, round(6 * scale)),
    )

    left, top, right, bottom = main_bounds
    inset = round(28 * scale)
    artwork_bounds = (left + inset, top + inset, right - inset, bottom - inset)
    art = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    art_draw = ImageDraw.Draw(art, "RGBA")
    art_draw.rounded_rectangle(
        artwork_bounds,
        radius=round(62 * scale),
        fill=_hex("#f5ecdd"),
    )
    ax1, ay1, ax2, ay2 = artwork_bounds
    sky = _linear_gradient(max(1, ax2 - ax1), max(1, ay2 - ay1), "#f6d39b", "#f0a169")
    art.alpha_composite(sky, (ax1, ay1))
    hill = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    hill_draw = ImageDraw.Draw(hill, "RGBA")
    hill_draw.pieslice(
        (
            round((ax1 - 60 * scale)),
            round((ay2 - 210 * scale)),
            round((ax2 + 120 * scale)),
            round((ay2 + 160 * scale)),
        ),
        start=188,
        end=360,
        fill=_hex("#245f57"),
    )
    hill_draw.pieslice(
        (
            round((ax1 + 80 * scale)),
            round((ay2 - 160 * scale)),
            round((ax2 + 170 * scale)),
            round((ay2 + 200 * scale)),
        ),
        start=188,
        end=360,
        fill=_hex("#4d8d7f"),
    )
    hill_draw.ellipse(
        (
            round((ax1 + 80 * scale)),
            round((ay1 + 62 * scale)),
            round((ax1 + 190 * scale)),
            round((ay1 + 172 * scale)),
        ),
        fill=_hex("#fff5dc", 210),
    )
    art.alpha_composite(hill)

    pin_color = _hex("#245f57", 210)
    dot_radius = max(4, round(6 * scale))
    for dot_x, dot_y in ((560, 370), (600, 430), (510, 460), (610, 520)):
        cx = round(dot_x * scale)
        cy = round(dot_y * scale)
        art_draw.ellipse((cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius), fill=pin_color)

    card.alpha_composite(art)
    canvas.alpha_composite(card)


def _draw_magnifier(canvas: Image.Image, scale: float) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    lens_center = (round(688 * scale), round(646 * scale))
    outer_radius = round(170 * scale)
    inner_radius = round(130 * scale)
    ring_width = max(8, round(22 * scale))

    ring_bounds = (
        lens_center[0] - outer_radius,
        lens_center[1] - outer_radius,
        lens_center[0] + outer_radius,
        lens_center[1] + outer_radius,
    )
    draw.ellipse(ring_bounds, fill=_hex("#1f5651"))
    highlight_bounds = (
        lens_center[0] - inner_radius,
        lens_center[1] - inner_radius,
        lens_center[0] + inner_radius,
        lens_center[1] + inner_radius,
    )
    draw.ellipse(highlight_bounds, fill=_hex("#93d3bd", 130), outline=_hex("#dff7ef", 180), width=max(5, round(8 * scale)))
    draw.ellipse(
        (
            lens_center[0] - round(86 * scale),
            lens_center[1] - round(104 * scale),
            lens_center[0] + round(12 * scale),
            lens_center[1] - round(12 * scale),
        ),
        fill=_hex("#ffffff", 52),
    )

    grid_color = _hex("#245f57", 178)
    cell = round(48 * scale)
    radius = max(4, round(6 * scale))
    for row in range(3):
        for col in range(3):
            cx = round((614 + col * 52) * scale)
            cy = round((574 + row * 52) * scale)
            draw.rounded_rectangle(
                (
                    cx - round(cell * 0.36),
                    cy - round(cell * 0.36),
                    cx + round(cell * 0.36),
                    cy + round(cell * 0.36),
                ),
                radius=radius,
                outline=grid_color,
                width=max(4, round(6 * scale)),
            )

    handle = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    handle_draw = ImageDraw.Draw(handle, "RGBA")
    handle_draw.rounded_rectangle(
        (
            round(720 * scale),
            round(742 * scale),
            round(892 * scale),
            round(826 * scale),
        ),
        radius=round(36 * scale),
        fill=_hex("#ca8858"),
    )
    handle_draw.rounded_rectangle(
        (
            round(744 * scale),
            round(758 * scale),
            round(882 * scale),
            round(794 * scale),
        ),
        radius=round(18 * scale),
        fill=_hex("#f4cb91", 165),
    )
    rotated = handle.rotate(-38, resample=Image.Resampling.BICUBIC)
    canvas.alpha_composite(rotated)

    draw.ellipse(
        (
            lens_center[0] - outer_radius + ring_width,
            lens_center[1] - outer_radius + ring_width,
            lens_center[0] + outer_radius - ring_width,
            lens_center[1] + outer_radius - ring_width,
        ),
        outline=_hex("#f4efe3", 120),
        width=max(4, round(8 * scale)),
    )


def render_icon(size: int = 1024) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    scale = size / 1024
    margin = round(64 * scale)
    radius = round(228 * scale)
    mask = _rounded_rect_mask(size, margin, radius)

    background = _linear_gradient(size, size, "#f6f0e3", "#d8e7df")
    background = Image.alpha_composite(
        background,
        _ellipse_overlay(size, (round(60 * scale), round(48 * scale), round(478 * scale), round(412 * scale)), "#d7a76b", 72, 34 * scale),
    )
    background = Image.alpha_composite(
        background,
        _ellipse_overlay(size, (round(508 * scale), round(542 * scale), round(964 * scale), round(944 * scale)), "#2f7f70", 86, 42 * scale),
    )
    shell = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shell.paste(background, mask=mask)
    canvas.alpha_composite(shell)

    outline = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    outline_draw = ImageDraw.Draw(outline, "RGBA")
    outline_draw.rounded_rectangle(
        (margin, margin, size - margin, size - margin),
        radius=radius,
        outline=_hex("#c9baa0", 165),
        width=max(3, round(6 * scale)),
    )
    canvas.alpha_composite(outline)

    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    shadow_draw.rounded_rectangle(
        (
            round(214 * scale),
            round(216 * scale),
            round(784 * scale),
            round(804 * scale),
        ),
        radius=round(88 * scale),
        fill=_hex("#463a28", 42),
    )
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(30 * scale)))

    _draw_photo_card(canvas, scale)
    _draw_magnifier(canvas, scale)

    return canvas


def build_icons(output_dir: Path = ICON_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = render_icon(1024)
    written: list[Path] = []

    for filename, size in EXPORT_LAYOUT.items():
        destination = output_dir / filename
        image = base if size == 1024 else base.resize((size, size), Image.Resampling.LANCZOS)
        image.save(destination)
        written.append(destination)

    with tempfile.TemporaryDirectory(prefix="semanticgallery-iconset-") as temp_dir:
        iconset_dir = Path(temp_dir) / "SemanticGallery.iconset"
        iconset_dir.mkdir()
        for filename, size in ICONSET_LAYOUT.items():
            image = base if size == 1024 else base.resize((size, size), Image.Resampling.LANCZOS)
            image.save(iconset_dir / filename)
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(output_dir / "icon.icns")],
            check=True,
        )
    written.append(output_dir / "icon.icns")

    return written


def main() -> None:
    written = build_icons()
    for path in written:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
