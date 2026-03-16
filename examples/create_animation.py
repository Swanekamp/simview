"""
Example showing how to generate frames and build an animation.
"""

from simview.make_gif import make_gif

make_gif(
    frame_dir="frames",
    output="simulation.gif",
    fps=10
)


