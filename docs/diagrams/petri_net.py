"""Petri net of the conveyor in the guide's notation (table 3.2).

python -m docs.diagrams.petri_net

Place - circle with its initial marking, transition - thin bar with the name
above and the delay d and «r, b» (priority, probability) below, arc weight k -
a tick across the arc with the number (weight 1 is not shown).
The figure is drawn at its print size (16.5 cm wide), so the font sizes below
are the sizes on the page.
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})

CM = 1 / 2.54
WIDTH_CM = 16.5
R = 0.32                # place radius
BW, BH = 0.16, 0.95     # transition bar
FS_NAME, FS_PARAM, FS_PLACE, FS_WEIGHT = 9, 8.5, 10, 8.5


class Drawing:
    def __init__(self, ax):
        self.ax = ax
        self.nodes = {}

    def place(self, key, x, y, label, markers="", label_pos="top"):
        self.ax.add_patch(Circle((x, y), R, fill=False, lw=1.1))
        if markers != "":
            self.ax.text(x, y, str(markers), ha="center", va="center", fontsize=FS_PLACE)
        dx, dy, ha, va = {"top": (0, R + 0.12, "center", "bottom"),
                          "bottom": (0, -R - 0.12, "center", "top"),
                          "left": (-R - 0.1, 0, "right", "center"),
                          "right": (R + 0.1, 0, "left", "center")}[label_pos]
        self.ax.text(x + dx, y + dy, label, ha=ha, va=va, fontsize=FS_PLACE)
        self.nodes[key] = ("place", x, y)

    def transition(self, key, x, y, name, delay, r=0, b=1.0, name_pos="top"):
        self.ax.add_patch(Rectangle((x - BW / 2, y - BH / 2), BW, BH, fill=False, lw=1.1))
        if name_pos == "top":
            self.ax.text(x, y + BH / 2 + 0.12, name, ha="center", va="bottom", fontsize=FS_NAME)
        elif name_pos == "topleft":
            self.ax.text(x - 0.1, y + BH / 2 + 0.12, name, ha="right", va="bottom", fontsize=FS_NAME)
        else:  # to the upper right of the bar
            self.ax.text(x + BW / 2 + 0.1, y + BH / 2, name, ha="left", va="bottom", fontsize=FS_NAME)
        self.ax.text(x, y - BH / 2 - 0.1, f"{delay}\n{r}, {b:.1f}", ha="center", va="top",
                     fontsize=FS_PARAM, linespacing=1.15)
        self.nodes[key] = ("trans", x, y)

    def _edge_point(self, key, tx, ty):
        kind, x, y = self.nodes[key]
        dx, dy = tx - x, ty - y
        n = math.hypot(dx, dy)
        if kind == "place":
            return x + R * dx / n, y + R * dy / n
        sx = (BW / 2) / abs(dx) if dx else math.inf
        sy = (BH / 2) / abs(dy) if dy else math.inf
        s = min(sx, sy)
        return x + dx * s, y + dy * s

    def arc(self, src, dst, weight=1, offset=0.0, rad=0.0, tick_at=0.45, label_side=1, end=None):
        _, x1, y1 = self.nodes[src]
        _, x2, y2 = self.nodes[dst]
        # parallel offset (for the two arcs of a self-loop)
        nx, ny = -(y2 - y1), x2 - x1
        n = math.hypot(nx, ny)
        ox, oy = offset * nx / n, offset * ny / n
        p1 = self._edge_point(src, x2 + ox, y2 + oy)
        p2 = self._edge_point(dst, x1 + ox, y1 + oy)
        p1 = (p1[0] + ox, p1[1] + oy)
        p2 = end if end is not None else (p2[0] + ox, p2[1] + oy)
        self.ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                          color="black", connectionstyle=f"arc3,rad={rad}",
                                          shrinkA=0, shrinkB=0))
        if weight != 1:
            # point on the (possibly curved) arc: quadratic Bezier of arc3
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            cx, cy = mx + rad * dy, my - rad * dx
            t = tick_at
            px = (1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * cx + t ** 2 * p2[0]
            py = (1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * cy + t ** 2 * p2[1]
            tx = 2 * (1 - t) * (cx - p1[0]) + 2 * t * (p2[0] - cx)
            ty = 2 * (1 - t) * (cy - p1[1]) + 2 * t * (p2[1] - cy)
            tn = math.hypot(tx, ty)
            ux, uy = tx / tn, ty / tn          # along the arc
            vx, vy = -uy, ux                   # across the arc
            # tick: across the arc, tilted like «/»
            h = 0.13
            ax_, ay_ = vx * 0.9 + ux * 0.45, vy * 0.9 + uy * 0.45
            self.ax.plot([px - h * ax_, px + h * ax_], [py - h * ay_, py + h * ay_], color="black", lw=0.9)
            self.ax.text(px + label_side * 0.26 * vx, py + label_side * 0.26 * vy, str(weight),
                         ha="center", va="center", fontsize=FS_WEIGHT)


def draw(buffer=None, path=None):
    has_buf = buffer is not None
    x_lo, x_hi = -0.3, 12.3
    y_lo, y_hi = (-3.3, 7.4) if has_buf else (-2.6, 5.4)
    fig_w = WIDTH_CM * CM
    fig = plt.figure(figsize=(fig_w, fig_w * (y_hi - y_lo) / (x_hi - x_lo)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.axis("off")
    d = Drawing(ax)

    y1, y2, y3 = (5.2, 2.6, -0.4) if has_buf else (4.2, 2.2, 0.2)
    x_g, x_a, x_q, x_f, x_n = 0.6, 2.8, 5.2, 8.0, 10.6

    d.place("G1", x_g, y1, "$G_1$", 1, "top")
    d.place("G2", x_g, y2, "$G_2$", 1, "top")
    d.place("C", x_g, y3, "$C$", 1, "top")
    d.transition("A1", x_a, y1, "Надходження 1", "unif(5, 1)",
                 name_pos="topleft" if has_buf else "top")
    d.transition("A2", x_a, y2, "Надходження 2", "unif(20, 7)")
    d.transition("TC", x_a, y3, "Рух конвеєра", "10")
    d.place("Q1", x_q, y1, "$Q_1$", "", "bottom" if has_buf else "top")
    d.place("Q2", x_q, y2, "$Q_2$", "", "top")
    d.place("S", x_q, y3, "$S$", "", "bottom")
    d.transition("FILL", x_f, y2, "Комплектація", "0", r=1,
                 name_pos="right" if has_buf else "top")
    d.transition("SKIP", x_f, y3 - 1.3, "Пропуск секції", "0", r=0)
    d.place("NF", x_n, y2, "$N_{fill}$", "", "right")
    d.place("NS", x_n, y3 - 1.3, "$N_{skip}$", "", "right")

    for g, a in (("G1", "A1"), ("G2", "A2"), ("C", "TC")):
        d.arc(g, a, offset=0.14)
        d.arc(a, g, offset=0.14)
    d.arc("A1", "Q1", 5)
    d.arc("A2", "Q2", 20)
    d.arc("TC", "S")
    d.arc("Q1", "FILL", 10, tick_at=0.4)
    d.arc("Q2", "FILL", 10, tick_at=0.4)
    d.arc("S", "FILL")
    d.arc("S", "SKIP")
    d.arc("FILL", "NF")
    d.arc("SKIP", "NS")

    if has_buf:
        l1, l2 = buffer
        d.place("F1", x_f, y1, "$F_1$", l1, "right")
        d.place("F2", x_q, (y2 + y3) / 2, "$F_2$", l2, "left")
        d.arc("F1", "A1", 5, rad=0.35, tick_at=0.5, label_side=1,
              end=(x_a, y1 + BH / 2))
        d.arc("FILL", "F1", 10, tick_at=0.5, label_side=-1)
        d.arc("F2", "A2", 20, tick_at=0.5, label_side=-1)
        d.arc("FILL", "F2", 10, tick_at=0.5)

    fig.savefig(path, dpi=400)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main():
    draw(None, OUT / "petri_net_base.png")
    draw((20, 60), OUT / "petri_net_buffer.png")


if __name__ == "__main__":
    main()
