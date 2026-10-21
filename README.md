# SAVE: Search-Aware Value-Based Early Stopping

Code for the paper *SAVE: Search-Aware Value-Based Early Stopping for Hyperparameter Optimization in Deep Learning* (TODO: link and DOI).

SAVE decides, after every epoch of a run in a sequential hyperparameter search, whether to continue it. It forecasts the run from the learning curves of the earlier runs of the same search, including runs that were stopped early (donors are chosen by similarity in level and recent slope and their
increments are stitched together), and continues the run while the expected gain of the incumbent, $max_k [G(k) − c (k − t)]$, is positive. Its only tuned constant is the price of a full training, $cT = 7.9 × 10⁻⁴$ It needs no pretraining and runs on a CPU.

The repository contains the method, the search setting (sequential search, budget of 20 full trainings, final phase), the baselines and oracles, the evaluation (speed-up, failures, bootstrap intervals, paired differences), the selection of settings, the data splits and the recorded NinaPro DB5 learning curves. The selected settings of every method are in `configs/`.

## Installation

Python ≥ 3.10.

```bash
pip install -e .
```

LC-PFN and FT-PFN are optional (`pip install -e .[pfn]`). The LC-PFN tables of the paper were computed with torch 1.11, lcpfn 0.1.3 and TF32 off, FT-PFN with ifbo 0.4.1 and its released checkpoint `bopfn_broken_unisep_1000curves_10params_2M.pt`.

## Data

```bash
python scripts/download_data.py nb201 lcbench
```

downloads NAS-Bench-201 (the NATS-Bench topology records, 1.1 GB) and LCBench (`data_2k_lw.zip`, 340 MB) into `data/raw/`, checks their checksums and writes the arrays the experiments read (`data/nb201_<task>.npz`, `data/lcbench/<dataset>.npz`). Set `SAVE_DATA` to keep them elsewhere.

Splits:

- NAS-Bench-201: per task, 1,000 development, 2,000 preliminary and 12,625 test architectures, listed in `splits/nb201/<task>_{dev,prelim,test}.txt` (indices into the 15,625 architectures). The curves are those of seed 777 at every second epoch, so T = 100 grid steps.
- LCBench: configurations 0–1199 of every dataset are the development part, 1200–1999 the test part.
- NinaPro DB5: all of it is test data. `data/db5_curves.csv.gz` holds the curves we recorded (10 subjects × 2 learning-rate schedules × 60 configurations × 50 epochs; validation and test accuracy, training loss), and `data/db5_subjects.csv` holds the window counts (the recording resolution is 1 / number of validation windows). The 60 configurations are in `configs/db5_space.yaml`.

**NinaPro DB5.** If you use the DB5 curves, cite Pizzolato et al. (2017), *Comparison of six electromyography acquisition setups on hand movement classification tasks*, PLoS ONE 12(10): e0186132. The raw signals are not included: their licence (CC BY-ND 4.0) does not allow distributing
derived data, so `scripts/db5_train.py` downloads them from Zenodo itself.

Random orders, pools, bootstrap replicates and forecast noise are drawn from the seeds in `configs/seeds.json`.

## Methods

`scripts/run.py` simulates one method on one benchmark (every unit, 40 random orders) and writes the stop epochs and the incumbent after every epoch to `results/`. `scripts/evaluate.py` turns these into speed-ups, saved fractions, failures, 95 % bootstrap intervals and paired differences. Every evaluation needs the reference `no-stop` in the same frame.

| method | |
|---|---|
| `save` | SAVE |
| `save-threshold` | ablation: the SAVE forecast with a threshold rule instead of the value rule |
| `save-completed-donors` | ablation: SAVE with donors trained to completion only |
| `save-multiplicative-uniform` | the alternative form of the forecast: multiplicative continuation, uniform donors |
| `median`, `successive-halving`, `hyperband`, `asha` | rank-based schedulers |
| `cl2017`, `cl2017-value` | Chandrashekaran & Lane with its own rule / with the value rule |
| `lcpfn-threshold`, `lcpfn-value` | LC-PFN with its threshold rule / with the value rule |
| `parametric-value`, `parametric-threshold` | parametric forecast (exponential model, SMC) with the value rule / a threshold rule |
| `ftpfn-threshold`, `ftpfn-value` | FT-PFN in the stopping rule |
| `fixed-budget`, `fixed-budget-tmin` | i epochs per configuration |
| `oracle-perfect-forecast`, `oracle-best-on-average`, `oracle-all-knowing` | oracle ceilings |
| `no-stop` | search without early stopping |

SAVE, C&L, ASHA and Hyperband train the first run of a search to T. Median stopping stops it only by its patience rule. The rules with LC-PFN and with the parametric forecast decide about it as about any other run.

## Settings and how they were selected

`configs/<benchmark>.yaml` holds the selected settings (LCBench: a default and the exceptions per dataset); `configs/grids.yaml` holds the grids they were selected from. The rule: among the settings that fail on no selection unit, the one with the highest mean speed-up; if every setting fails somewhere, the one with the fewest failing units and then the highest mean speed-up with a failure counted as 1; ties go to the more cautious setting.

- NAS-Bench-201: selected on the development architectures of the three tasks.
- LCBench: selected for every dataset on the test configurations of the other 34 datasets. Median stopping uses one setting selected on all 35 datasets, and fixed-budget evaluation uses 5 epochs without selection. C&L uses the threshold selected for `parametric-threshold`, and C&L with the value rule uses the price of SAVE.
- NinaPro DB5: the LCBench settings selected on all 35 datasets at once, on all their configurations.
- SAVE's price and the threshold of `save-threshold` (one value for all benchmarks): the value whose worst ratio of speed-up to that of median stopping over the development data of LCBench and of the three NAS-Bench-201 tasks is largest.
- FT-PFN was not selected on NAS-Bench-201 and DB5 (it uses the LC-PFN threshold and price of DB5); on LCBench its threshold is selected per dataset like the others, on the first three orders.

```bash
python scripts/select_settings.py --criterion worst-ratio --method save --jobs 8
python scripts/select_settings.py --criterion worst-ratio --method save-threshold --jobs 8
python scripts/select_settings.py --benchmark nb201 --method asha --jobs 8
python scripts/select_settings.py --benchmark lcbench --method asha --jobs 8
python scripts/select_settings.py --benchmark lcbench --method asha --all-datasets --jobs 8
```

The methods with decision tables need the tables of the whole grid first, for example
`python scripts/tables.py --benchmark nb201 --source lcpfn --split dev --grid`.

## Running the experiments

### Comparison of methods

```bash
for m in no-stop save save-threshold median successive-halving hyperband asha cl2017 fixed-budget fixed-budget-tmin; do
  python scripts/run.py --benchmark nb201 --method $m --jobs 8
done
python scripts/evaluate.py --benchmark nb201 --methods no-stop save save-threshold median successive-halving hyperband asha cl2017 fixed-budget --paired save
```

The same works with `--benchmark lcbench` and `--benchmark db5`. DB5 runs with the final phase by
default; `--no-final-phase` gives the results without it, evaluated with `--frame main`. On LCBench
`evaluate.py` prints the datasets on which a method fails. `--benchmark nb201 lcbench db5` also reports
each method's speed-up as a share of the best method's speed-up on every test set. `--curves file.csv` writes the mean regret curve of every method and test set.

LC-PFN and the parametric forecast are evaluated once per curve and decision epoch. Make their decision tables first:

```bash
python scripts/tables.py --benchmark lcbench --source parametric --jobs 16
python scripts/tables.py --benchmark lcbench --source lcpfn --device cuda
python scripts/run.py --benchmark lcbench --method parametric-value
python scripts/run.py --benchmark lcbench --method lcpfn-value
```

FT-PFN runs inside the search and is slow. It was evaluated on the first five orders (three on LCBench), and the comparison methods must then be run on the same orders:

```bash
python scripts/run.py --benchmark db5 --method ftpfn-threshold --set delta=0.1 --first-orders 5 \
    --ftpfn-checkpoint path/to/bopfn_broken_unisep_1000curves_10params_2M.pt
```

### Oracles

```bash
python scripts/oracles.py --benchmark nb201 --jobs 3
```

runs the all-knowing, best-on-average and perfect-forecast oracles and prints their speed-ups and the split of SAVE's gap to the best-on-average oracle into the part due to the forecast and the part due to the rule. It needs `no-stop` and `save` of the same frame. The dynamic programme of the best-on-average oracle takes 15–35 minutes per NAS-Bench-201 task. The perfect-forecast oracle runs at any price:

```bash
python scripts/run.py --benchmark nb201 --method oracle-perfect-forecast --set price=0.0
python scripts/run.py --benchmark nb201 --method oracle-perfect-forecast --price-sensitivity
```

### Size of the pool

```bash
python scripts/pool_size.py --benchmark lcbench --methods save median asha save-threshold --jobs 8
```

searches over 40 random pools of P configurations (P = 60, 120, 250, 500, and 1000 on NAS-Bench-201), with and without the final phase, and prints the number of failing units per P.

### Calibration of the forecast

```bash
python scripts/calibration.py --benchmark nb201 lcbench db5
```

prints the PIT of the final best accuracy at the first decision point, how often the winner lies above the 80 % interval and how often SAVE keeps it, in searches over 60 configurations.

### Sensitivity to the price and to the number of paths

```bash
python scripts/run.py --benchmark lcbench --method save --price-sensitivity --jobs 8
python scripts/run.py --benchmark lcbench --method save --set paths=100 --jobs 8
python scripts/evaluate.py --benchmark lcbench --methods no-stop save save@price=0.0 save@paths=100
```

`--price-sensitivity` runs a method at every price of `price_sensitivity` in
`configs/<benchmark>.yaml` (per full training, from 0 to 6.3 times the price of SAVE). A result label is the method name followed by its overridden settings, as printed by `run.py`.

### Characteristics of a benchmark

```bash
python scripts/characteristics.py --benchmark nb201 lcbench db5
```

reports, per unit and as the median over a set:
- the share of a curve's gain obtained in the last quarter of training;
- the fraction of training after which 95 % of the gain is reached;
- the Spearman correlation between the accuracy at the first decision point and the final accuracy;
- the share of late winners in searches over 60 configurations.

### Recording the DB5 curves again

```bash
pip install -e .[db5]
python scripts/db5_train.py --raw data/raw/db5 --out db5_curves_new.csv.gz
```

downloads the ten subjects from Zenodo (record 1000116), prepares the windows and features, and trains the 60 configurations under both schedules, which takes about a day on a laptop CPU. On the CPU it reproduces the curves in `data/`.

The experiment scripts accept `--units` to restrict them to some units, and `run.py` accepts `--orders` and `--first-orders`; `db5_train.py` accepts `--configs` and `--epochs`. This is useful for a quick look.

## Layout

```
save/forecast.py       the SAVE forecast (donors, stitching, noise, M paths)
save/rule.py           value rule, threshold rule, decision tables
save/method.py         SAVE and the perfect-forecast oracle as stopping policies
save/search.py         sequential search, budget, final phase
save/baselines/        median stopping, SH, Hyperband, ASHA, fixed budget, C&L, LC-PFN, FT-PFN, parametric
save/oracles.py        all-knowing envelope, best-on-average dynamic programme
save/metrics.py        speed-up, saved fraction, failures, bootstrap panels, paired differences
save/data/             NAS-Bench-201, LCBench and DB5 loaders
configs/               selected settings, grids, seeds, DB5 configurations
splits/                NAS-Bench-201 split of the architectures
scripts/               command-line entry points
```

## Licence

Code: MIT. The DB5 learning curves are released with the code; the NinaPro DB5 data themselves remain under their own licence (CC BY-ND 4.0).
