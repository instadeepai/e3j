"""Regenerates the README poster frame from the animation itself: fast-forwards
`fwd_anim` to its `snapshot()` hook and saves that frame. Run it whenever the
animation changes; the committed PNG is its output.

    uvx --with manim==0.20.1 python docs/animations/make_thumbnail.py
"""

import sys
from pathlib import Path

from manim import BLACK, PI, RIGHT, WHITE, Circle, Triangle, VGroup, config
from manim.utils.exceptions import EndSceneEarlyException
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from fwd_anim import StreamingMessagePassing  # noqa: E402

OUT = Path(__file__).parent / "kernel_thumbnail.png"
FRAME = "first-flush"
SIZE = (540, 304)  # the width the README displays it at


def play_button():
    ring = Circle(
        radius=0.95,
        stroke_color=WHITE,
        stroke_width=9,
        fill_color=BLACK,
        fill_opacity=0.35,
    )
    tri = Triangle(color=WHITE, fill_color=WHITE, fill_opacity=1)
    tri.scale(0.45).rotate(-PI / 2).move_to(ring)
    tri.shift(tri.width / 6 * RIGHT)
    return VGroup(ring, tri)


class Thumbnail(StreamingMessagePassing):

    def snapshot(self, name):
        if name == FRAME:
            self.add(play_button())
            self.renderer.update_frame(self, self.mobjects)
            frame = self.camera.get_image().convert("RGB")
            # Palette-quantized: flat colour and hard-edged text index well.
            frame.convert("P", palette=Image.ADAPTIVE).save(OUT, optimize=True)
            raise EndSceneEarlyException()


if __name__ == "__main__":
    config.pixel_width, config.pixel_height = SIZE
    config.dry_run = True  # no manim output files, no ffmpeg
    config.progress_bar = "none"
    config.verbosity = "WARNING"
    OUT.unlink(missing_ok=True)
    Thumbnail(skip_animations=True).render()
    assert OUT.exists(), f"animation never reached snapshot({FRAME!r})"
    print(f"{OUT}  ({OUT.stat().st_size / 1024:.0f} KiB)")
