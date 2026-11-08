"""Process of the completing conveyor, BPMN-style (base scheme).

python -m docs.diagrams.process

Thin circle - start event, bold circle - end event, rounded box - task,
diamond - exclusive gateway, cylinder - data store (accumulator),
solid arrow - sequence flow, dashed arrow - data flow.
Drawn at print size: 16.5 cm wide, coordinates below are in centimetres.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch,  # noqa: E402
                                FancyBboxPatch, Polygon, Rectangle)

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})
CM = 1 / 2.54
W, H = 16.5, 13.6
FS, FS_SMALL, FS_LANE = 9, 8, 9.5


def task(ax, x, y, text, w=3.0, h=1.05):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.25",
                                fill=False, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, linespacing=1.15)
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}


def event(ax, x, y, end=False, r=0.28):
    ax.add_patch(Circle((x, y), r, fill=False, lw=2.2 if end else 0.9))
    return {"l": (x - r, y), "r": (x + r, y), "t": (x, y + r), "b": (x, y - r)}


def gateway(ax, x, y, text, w=3.0, h=1.5):
    ax.add_patch(Polygon([(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)],
                         closed=True, fill=False, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, linespacing=1.1)
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}


def store(ax, x, y, text, w=2.6, h=1.2, e=0.32):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False, lw=0, ec="none"))
    ax.plot([x - w / 2, x - w / 2], [y - h / 2, y + h / 2], color="k", lw=0.9)
    ax.plot([x + w / 2, x + w / 2], [y - h / 2, y + h / 2], color="k", lw=0.9)
    ax.add_patch(Ellipse((x, y + h / 2), w, e, fill=False, lw=0.9))
    ax.add_patch(matplotlib.patches.Arc((x, y - h / 2), w, e, theta1=180, theta2=360, lw=0.9))
    ax.text(x, y - 0.05, text, ha="center", va="center", fontsize=FS, linespacing=1.1)
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2 + e / 2), "b": (x, y - h / 2 - e / 2)}


def flow(ax, p1, p2, dashed=False, rad=0.0, label=None, label_xy=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>" if not dashed else "->", mutation_scale=9,
                                 lw=0.9, color="k", ls="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=FS_SMALL)


def lane(ax, y0, y1, title):
    ax.add_patch(FancyBboxPatch((0.1, y0), W - 0.2, y1 - y0, boxstyle="round,pad=0,rounding_size=0.25",
                                fill=False, lw=0.7, ec="0.35"))
    ax.text(0.35, y1 - 0.2, title, ha="left", va="top", fontsize=FS_LANE, style="italic")


def supplier(ax, y_task, y0, y1, title, batch_text, top, store_xy, store_text):
    """top=True: lane above the conveyor (loop above the tasks, data flow down)."""
    lane(ax, y0, y1, title)
    s = event(ax, 1.0, y_task)
    p = task(ax, 5.3, y_task, batch_text, w=3.6)
    d = task(ax, 11.1, y_task, "Передача партії\nу накопичувач", w=2.8)
    flow(ax, s["r"], p["l"])
    flow(ax, p["r"], d["l"])
    if top:
        flow(ax, d["t"], p["t"], rad=0.3)
        ax.text(8.2, y_task + 1.05, "наступна партія", ha="center", va="center", fontsize=FS_SMALL)
    else:
        flow(ax, d["b"], p["b"], rad=-0.3)
        ax.text(8.2, y_task - 1.05, "наступна партія", ha="center", va="center", fontsize=FS_SMALL)
    q = store(ax, *store_xy, store_text)
    flow(ax, d["b"], q["t"], dashed=True) if top else flow(ax, d["t"], q["b"], dashed=True)
    return q


def main():
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    y_lo = -1.85
    ax.set_xlim(0, W)
    ax.set_ylim(y_lo, y_lo + H)
    ax.set_aspect("equal")
    ax.axis("off")

    q1 = supplier(ax, 9.9, 8.8, 11.65, "Постачальник виробів типу 1",
                  "Виготовлення партії\n5 виробів, unif(5, 1) хв", True, (11.1, 7.75),
                  "Накопичувач\nвиробів типу 1\n($Q_1$)")
    q2 = supplier(ax, -0.15, -1.75, 1.15, "Постачальник виробів типу 2",
                  "Виготовлення партії\n20 виробів, unif(20, 7) хв", False, (11.1, 1.95),
                  "Накопичувач\nвиробів типу 2\n($Q_2$)")

    lane(ax, 2.85, 6.7, "Комплектувальний конвеєр")
    s = event(ax, 1.0, 4.9)
    mv = task(ax, 3.55, 4.9, "Подача порожньої\nсекції до поста\n(кожні 10 хв)", w=2.7, h=1.25)
    gw = gateway(ax, 6.7, 4.9, "Є ≥ 10 виробів\nкожного типу?", w=3.0, h=1.6)
    fill = task(ax, 11.1, 4.9, "Комплектація секції:\nвзяти по 10 виробів\nкожного типу", w=3.1, h=1.25)
    e1 = event(ax, 14.6, 4.9, end=True)
    ax.text(14.6, 4.45, "секцію\nзаповнено", ha="center", va="top", fontsize=FS_SMALL)
    skip = task(ax, 6.6, 3.4, "Пропуск: секція\nйде порожньою", w=2.6, h=0.75)
    e2 = event(ax, 8.8, 3.4, end=True)
    ax.text(8.8, 3.72, "секцію\nпропущено", ha="center", va="bottom", fontsize=FS_SMALL)

    flow(ax, s["r"], mv["l"])
    flow(ax, mv["r"], gw["l"])
    flow(ax, gw["r"], fill["l"])
    ax.text(8.6, 5.1, "так", ha="center", va="bottom", fontsize=FS_SMALL)
    flow(ax, gw["b"], (gw["b"][0], skip["t"][1]))
    ax.text(6.85, 3.92, "ні", ha="left", va="center", fontsize=FS_SMALL)
    flow(ax, fill["r"], e1["l"])
    flow(ax, skip["r"], e2["l"])
    # data flow: the filling takes 10 items from each accumulator
    flow(ax, q1["b"], fill["t"], dashed=True)
    flow(ax, q2["t"], fill["b"], dashed=True)
    ax.text(11.25, 6.25, "−10", ha="left", va="center", fontsize=FS_SMALL)
    ax.text(11.25, 3.5, "−10", ha="left", va="center", fontsize=FS_SMALL)

    fig.savefig(OUT / "process.png", dpi=400)
    fig.savefig(OUT / "process.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
