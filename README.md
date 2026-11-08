# Simulation model of an assembly conveyor (stochastic Petri net)

Coursework «Моделювання систем. Курсова робота», KPI, variant D2.

A completing conveyor receives batches of 5 type-1 items every 5±1 min and
20 type-2 items every 20±7 min. Every 10 min a conveyor section arrives; it is
filled if at least 10 items of each type are waiting, otherwise it leaves empty.
The model estimates the probability of an empty section, mean and maximum queue
length per item type, and compares the base scheme with 20 items of each type
per section at 1 section / 20 min.

The simulation engine (`src/petri`) is a self-written stochastic Petri net
simulator with time delays, priorities and multichannel transitions.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
pytest                                   # tests
```

## Layout

- `src/petri/` – Petri net simulation engine
- `src/model/` – conveyor model (base and 20/20 variant)
- `src/experiments/` – verification, transient period, replications, experiments
- `src/plots/` – charts
- `tests/` – pytest tests
- `results/` – CSV and PNG outputs
- `docs/diagrams/` – process diagram, Petri net, UML
- `report/` – coursework paper
