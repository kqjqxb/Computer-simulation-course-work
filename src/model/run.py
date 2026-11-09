"""Run the conveyor model from the command line.

Example:
    python -m src.model.run --k 10 --period 10 --T 1000 \\
        --seed 1 --runs 20 --trace
"""
import argparse

import numpy as np

from src.experiments.common import mean_ci, run_once
from src.petri import EventProtocol, const, unif

ROWS = [
    ("p_skip", "Ймовірність пропуску секції", "–", 4),
    ("n_fill", "Заповнено секцій", "од.", 1),
    ("n_skip", "Пропущено секцій", "од.", 1),
    ("q1_mean", "Середня довжина черги виробів типу 1", "од.", 2),
    ("q1_std", "СКВ довжини черги виробів типу 1", "од.", 2),
    ("q1_max", "Максимальна довжина черги виробів типу 1", "од.",
     1),
    ("q2_mean", "Середня довжина черги виробів типу 2", "од.", 2),
    ("q2_std", "СКВ довжини черги виробів типу 2", "од.", 2),
    ("q2_max", "Максимальна довжина черги виробів типу 2", "од.",
     1),
    ("sections_per_hour", "Пропускна здатність", "секцій/год", 3),
    ("items_per_hour", "Пропускна здатність (кожного типу)",
     "виробів/год", 2),
    ("w1", "Середній час очікування виробу типу 1", "хв", 2),
    ("w2", "Середній час очікування виробу типу 2", "хв", 2),
]
BUFFER_ROWS = [
    ("block1", "Частка часу блокування постачальника типу 1", "–",
     4),
    ("block2", "Частка часу блокування постачальника типу 2", "–",
     4),
    ("lost1", "Втрачене виробництво виробів типу 1",
     "виробів/год", 2),
    ("lost2", "Втрачене виробництво виробів типу 2",
     "виробів/год", 2),
]


def interval(mean, spread):
    return const(mean) if spread == 0 else unif(mean, spread)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Імітаційна модель "
                                            "комплектувального "
                                            "конвеєра (мережа "
                                            "Петрі)")
    p.add_argument("--k", type=int, default=10,
                   help="виробів кожного типу на секцію")
    p.add_argument("--period", type=float, default=10.0,
                   help="інтервал руху конвеєра, хв/секцію")
    p.add_argument("--T", type=float, default=1000.0,
                   help="час моделювання, хв")
    p.add_argument("--warmup", type=float, default=0.0,
                   help="початок збору статистики, хв")
    p.add_argument("--seed", type=int, default=1,
                   help="seed першого прогону (далі seed+1, ...)")
    p.add_argument("--runs", type=int, default=1,
                   help="кількість прогонів")
    p.add_argument("--batch1", type=int, default=5,
                   help="розмір партії виробів типу 1")
    p.add_argument("--int1", type=float, nargs=2,
                   default=(5.0, 1.0), metavar=("D", "S"),
                   help="інтервал надходження типу 1: unif(D, "
                        "S), хв; S=0 - детермінований")
    p.add_argument("--batch2", type=int, default=20,
                   help="розмір партії виробів типу 2")
    p.add_argument("--int2", type=float, nargs=2,
                   default=(20.0, 7.0), metavar=("D", "S"),
                   help="інтервал надходження типу 2: unif(D, "
                        "S), хв; S=0 - детермінований")
    p.add_argument("--q1-0", type=int, default=0,
                   help="початкова черга типу 1")
    p.add_argument("--q2-0", type=int, default=0,
                   help="початкова черга типу 2")
    p.add_argument("--buffer", type=int, nargs=2, default=None,
                   metavar=("L1", "L2"),
                   help="ємності буферів (канбан); без параметра "
                        "- базова модель")
    p.add_argument("--trace", action="store_true",
                   help="вивести протокол подій першого прогону")
    p.add_argument("--trace-limit", type=int, default=60,
                   help="кількість рядків протоколу")
    return p.parse_args(argv)


def fmt(x, digits):
    return "—" if x is None or np.isnan(x) else f"{x:.{digits}f}"


def main(argv=None):
    a = parse_args(argv)
    model = dict(k=a.k, period=a.period,
                 batch1=a.batch1, interval1=interval(*a.int1),
                 batch2=a.batch2, interval2=interval(*a.int2),
                 q1_0=a.q1_0, q2_0=a.q2_0,
                 buffer=tuple(a.buffer) if a.buffer else None)

    protocol = None
    if a.trace:
        places = ["Q1", "Q2", "S", "N_fill",
                  "N_skip"] + (["F1", "F2"] if a.buffer else [])
        protocol = EventProtocol(places=places,
                                 limit=a.trace_limit,
                                 headers=("t, хв", "фаза",
                                          "перехід"),
                                 phase_names={"IN": "вхід",
                                              "OUT": "вихід"})
    results = [run_once(a.seed + i, a.T, a.warmup, protocol
                        if i == 0 else None, **model)
               for i in range(a.runs)]
    rows = ROWS
    if a.buffer:
        for r in results:
            # supplier of 1 item/min idle
            r["lost1"] = 60.0 * r["block1"]
            r["lost2"] = 60.0 * r["block2"]
        rows = ROWS + BUFFER_ROWS

    print(f"Модель: k = {a.k} виробів/секцію, конвеєр 1 секція / "
          f"{a.period:g} хв; "
          f"тип 1: {a.batch1} шт. кожні {model['interval1']!r} "
          f"хв; "
          f"тип 2: {a.batch2} шт. кожні {model['interval2']!r} хв"
          + (f"; буфери L1 = {a.buffer[0]}, L2 = {a.buffer[1]}"
             if a.buffer else ""))
    print(f"Час моделювання T = {a.T:g} хв, збір статистики з "
          f"{a.warmup:g} хв, "
          f"прогонів: {a.runs}, seed = "
          f"{a.seed}..{a.seed + a.runs - 1}")
    print()

    name_w = max(len(f"{n}, {u}") for _, n, u, _ in rows)
    if a.runs == 1:
        print(f"{'Характеристика':<{name_w}}  {'Значення':>10}")
        for key, name, unit, d in rows:
            print(f"{name + ', ' + unit:<{name_w}}  "
                  f"{fmt(results[0][key], d):>10}")
    else:
        print(f"{'Характеристика':<{name_w}}  {'Середнє':>10}  "
              f"{'±95% ДІ':>9}  {'Мін':>9}  {'Макс':>9}")
        for key, name, unit, d in rows:
            vals = [r[key] for r in results]
            m, h = mean_ci(vals)
            print(f"{name + ', ' + unit:<{name_w}}  "
                  f"{fmt(m, d):>10}  {fmt(h, d):>9}  "
                  f"{fmt(min(vals), d):>9}  "
                  f"{fmt(max(vals), d):>9}")

    if protocol is not None:
        print()
        print(f"Протокол подій (прогін 1, перші "
              f"{len(protocol.rows)} подій):")
        print(protocol.to_text())


if __name__ == "__main__":
    main()
