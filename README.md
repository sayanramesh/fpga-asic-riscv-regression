# FPGA-to-ASIC Regression Pipeline for RISC-V Processor Cores

This repository contains the complete regression, classification and visualisation pipeline developed for the MSc dissertation *"Modelling the Relationship between FPGA and ASIC Implementations Using RISC-V Designs"* (University of Southampton, 2026).

The dissertation investigates whether FPGA implementation metrics (LUT count, maximum frequency, power, net count, flip-flop count, distributed RAM usage, unique control-set count) can predict corresponding ASIC implementation metrics (area, frequency, power, net count, register count), for fifteen architecturally diverse open-source RISC-V cores implemented on a Xilinx FPGA and two ASIC process nodes (ASAP7 7nm, FreePDK45 45nm), and vice versa.

## Repository structure

```
data/            Data loading utility and the full measured dataset
forward/         FPGA -> ASIC regression and classification pipeline
reverse/         ASIC -> FPGA regression and classification pipeline
model_averaging/ Cross-target, cross-combination model reliability ranking
```

## Requirements

```bash
pip install -r requirements.txt
```

## Running the pipeline

1. `data/load_data.py` provides the `load_metrics()` function used by every other script to read `metrics.csv`.
2. Run `forward/regression_analysis_v4.py` to reproduce the forward-direction regression sweep (11 models x 127 feature combinations x 10 targets x 2 fitting spaces).
3. Run `forward/classification_analysis.py` for the corresponding forward-direction classification sweep.
4. Run `reverse/regression_analysi_reverse_v4.py` and `reverse/classification_analysis_reverse.py` for the reverse direction.
5. Run `model_averaging/model_average.py` to reproduce the cross-target model reliability ranking.
6. Run the `visualise*.py` scripts in each directory to regenerate the summary tables, model-choice tables, model-ranking charts and cheap-combination Pareto-front figures used in the dissertation.

Each script writes its results to a CSV alongside a checkpoint file, and prints its progress per target so long-running sweeps do not appear to hang.

## Third-party RISC-V cores

This project measures, but does not redistribute, fifteen open-source RISC-V cores, each under its own separate license. See [THIRD_PARTY_CORES.md](THIRD_PARTY_CORES.md) for the full list, authors, and links to each core's original repository.

## License

The code in this repository (everything except the third-party cores referenced above) is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the full text.

## Citation

If you use this pipeline, please cite the accompanying dissertation. See `CITATION.cff` for citation metadata.
