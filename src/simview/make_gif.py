# simview/animate.py

from pathlib import Path
import imageio.v2 as imageio


def make_gif(frame_dir, output_name, fps=10, loop=0):
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("*.png"))
#    frames = sorted(frame_dir.glob("*.svg"))

    images = []
    for f in frames:
        images.append(imageio.imread(f))

    imageio.mimsave(output_name, images, fps=fps, loop=loop)

    print(f"GIF saved to {output_name}")