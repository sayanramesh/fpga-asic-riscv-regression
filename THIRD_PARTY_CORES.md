# Third-Party RISC-V Cores

This project implements and measures the following fifteen open-source RISC-V processor cores. **None of their source code is redistributed in this repository** — each core remains under its own original license, maintained in its own upstream repository. Only the implementation metrics measured from each core (area, frequency, power, net count, etc.) are included here, in `data/metrics.csv`.

| Core | Author / Organisation | Repository |
|---|---|---|
| SERV | Olof Kindgren | https://github.com/olofk/serv |
| FemtoRV32 | Bruno Levy | https://github.com/BrunoLevy/learn-fpga |
| PicoRV32 | Clifford Wolf | https://github.com/YosysHQ/picorv32 |
| DarkRISCV | darklife | https://github.com/darklife/darkriscv |
| Reindeer | PulseRain Technology | https://github.com/PulseRain/Reindeer |
| UltraEmbedded riscv | ultraembedded | https://github.com/ultraembedded/riscv |
| SimoDense | Philippos Papaphilippou | https://github.com/pphilippos/simodense |
| AngeloJacobo RISC-V | AngeloJacobo | https://github.com/AngeloJacobo/RISC-V |
| RudolV | bobbl | https://github.com/bobbl/rudolv |
| biRISC-V | ultraembedded | https://github.com/ultraembedded/biriscv |
| Engine-V (MF8A18) | micro-FPGA | https://github.com/micro-FPGA/engine-V |
| TinyRISCV | liangkangnan | https://github.com/liangkangnan/tinyriscv |
| RVX | Rafael Calçada | https://github.com/rafaelcalcada/rvx |
| Hazard3 | Luke Wren | https://github.com/Wren6991/Hazard3 |
| Glacial | Eric Smith | https://github.com/brouhaha/glacial |

If you wish to reproduce the FPGA and ASIC implementations from scratch, clone each core's own repository separately and follow its own build instructions; this repository's scripts operate on the resulting implementation reports, not on the RTL source itself.

Full citations for each core are provided in the dissertation's bibliography.
