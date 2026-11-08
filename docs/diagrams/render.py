"""Render all diagrams to PNG (and SVG).

python -m docs.diagrams.render

*.dot files (UML class diagram) are rendered with Graphviz; the process
diagram and the Petri nets are drawn by process.py and petri_net.py at their
print size (16.5 cm). For every Graphviz diagram the smallest font size on the page
is printed, assuming the picture is scaled to fit the text width (16.5 cm).
"""
import re
import subprocess
from pathlib import Path

from docs.diagrams import petri_net, process

HERE = Path(__file__).resolve().parent
TEXT_WIDTH_IN = 16.5 / 2.54
DPI = 300


def render_dot(src):
    for fmt in ("svg", "png"):
        args = ["dot", f"-T{fmt}", str(src), "-o", str(src.with_suffix("." + fmt))]
        if fmt == "png":
            args.insert(1, f"-Gdpi={DPI}")
        subprocess.run(args, check=True)
    svg = src.with_suffix(".svg").read_text(encoding="utf-8")
    width_pt = float(re.search(r'<svg width="([\d.]+)pt"', svg).group(1))
    sizes = [float(x) for x in re.findall(r'font-size="([\d.]+)"', svg)]
    scale = min(1.0, TEXT_WIDTH_IN * 72 / width_pt)
    return width_pt / 72, min(sizes) * scale


def main():
    for src in sorted(HERE.glob("*.dot")):
        width_in, min_font = render_dot(src)
        print(f"{src.name}: ширина {width_in * 2.54:.1f} см, найменший шрифт на сторінці {min_font:.1f} пт")
    process.main()
    print("process.png: ширина 16,5 см, найменший шрифт 8 пт")
    petri_net.main()
    print("petri_net_base.png, petri_net_buffer.png: ширина 16,5 см, найменший шрифт 8,5 пт")


if __name__ == "__main__":
    main()
