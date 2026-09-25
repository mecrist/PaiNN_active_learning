# PaiNN-AL: Active Learning Pipeline for Silica–Water Interfaces

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![NFF](https://img.shields.io/badge/NFF-Neural%20Force%20Field-green.svg)](https://github.com/learningmatter-mit/NeuralForceField)
[![ASE](https://img.shields.io/badge/ASE-3.26+-blueviolet.svg)](https://wiki.fysik.dtu.dk/ase/)
[![FHI-aims](https://img.shields.io/badge/DFT-FHI--aims-orange.svg)](https://fhi-aims.org/)

An autonomous, closed-loop active learning (AL) pipeline for Machine Learning Interatomic Potentials (MLIPs) using an ensemble of fine-tuned **PaiNN** (Polarizable Atom Interaction Neural Network) models coupled with **FHI-aims** quantum chemistry (DFT).

The pipeline explores complex mineral–water interfaces (alpha-quartz, beta-cristobalite, and amorphous silica with 5 Å, 10 Å, and 20 Å water layers), identifies out-of-distribution reactive/transition states using committee disagreement, queries FHI-aims single-point DFT, and fine-tunes the potential on the fly with zero data loss.

---

## 🔬 Key Features

- **6-Member Committee Ensemble**: Combines 6 fine-tuned PaiNN models (3 trained on internal datasets + 3 trained on merged datasets, epoch 500) to evaluate instantaneous model uncertainty via atomic force standard deviation.
- **Simultaneous 9-Geometry Exploration**: Drives 9 independent molecular dynamics trajectories concurrently at $T = 300\text{ K}$ using ASE Langevin dynamics across diverse polymorphs and water layer thicknesses.
- **Single-Node HPC Efficiency (1 GPU + 32 CPU Cores)**: Runs MD propagation and neural network inference on GPU while dispatching DFT single points directly to the allocated 32 CPU cores via local MPI (`mpirun -np 32`). Bypasses cluster queues completely.
- **Zero Data Loss DFT Archiving**: Every triggered FHI-aims calculation is permanently preserved in its own directory (`dft_calculations/step_{step}_traj_{idx}/`) containing `geometry.in`, `control.in`, and the full `aims.out`.
- **Reference Energy ($E_0$) Baseline Correction**: Automatically shifts raw all-electron DFT energies to zero-centered reference energies for PaiNN training, preventing gradient explosion.
- **Safe Online Retraining**: Retrains all 6 models for 1 epoch upon new data acquisition with custom NaN/Inf gradient guards (`make_safe_loss_fn` and `grad_is_finite`).
- **LAMMPS-Compatible Trajectory Dumps**:
  - **Unified Trajectory Dump** (`traj_traj{i}.lammpstrj`): Recorded every 1,000 steps (`id type mol xu yu zu`), with water vs. slab molecule tracking.
  - **Stress Dump** (`traj_stress_traj{i}.lammpstrj`): Recorded every 5,000 steps (`id type mol xu yu zu c_stress[1] c_stress[2] c_stress[3]`) for surface tension and virial analysis.
- **Walltime-Aware Checkpointing & Resume**: Tracks trajectory positions, velocities, step counts, and model weights in `al_checkpoint.json`. Traps Slurm `SIGTERM` signals before the 72-hour walltime to allow seamless 1-command resumption.

---

## 📐 Scientific Methodology

### 1. Committee Force Disagreement (Uncertainty Metric)
Following the exact formulation of [aims-PAX](https://github.com/tohenkes/aims-PAX/tree/main):
For an ensemble of $M$ models (3 or 6 members), each model $m$ predicts atomic forces $\mathbf{F}_m(i) \in \mathbb{R}^3$ on atom $i$.

- **Committee Mean Force**:
  $$\bar{\mathbf{F}}(i) = \frac{1}{M} \sum_{m=1}^M \mathbf{F}_m(i)$$

- **Atomic Force Standard Deviation (per atom, in eV/Å)**:
  $$\sigma_i = \sqrt{\frac{1}{3M} \sum_{m=1}^M \sum_{\alpha \in \{x,y,z\}} \left( F_{m,i,\alpha} - \bar{F}_{i,\alpha} \right)^2}$$

- **Global Configuration Uncertainty**:
  $$U = \max_{i \in \text{atoms}} \sigma_i \quad (\text{in eV/Å})$$

Taking the **maximum** atomic standard deviation rather than the mean ensures that localized rare events (e.g., surface silanol protonation, water dissociation, Si–O bond breaking) are detected immediately without being diluted across the remaining bulk/slab atoms.

### 2. Adaptive Rolling-Window Thresholding (aims-PAX Protocol)
- **Initial Threshold**: $U_{\text{thresh}} = \infty$ (burn-in period).
- **Burn-In Gate**: During the first 10 uncertainty evaluations ($250$ MD steps), no DFT calculations are triggered to allow thermalization and statistics accumulation.
- **Dynamic Rolling-Window Update**:
  $$U_{\text{thresh}} = \frac{1}{K} \sum_{k=1}^K U_{-k} \times (1 + c_x) \quad (K \le 400,\, c_x = 0.0)$$
  Once the training set reaches the target size ($N \ge 540$), the threshold is permanently frozen.

### 3. Elemental Reference Energy ($E_0$) Offsets
FHI-aims uses all-electron reference energies, yielding absolute energies of thousands of eV. PaiNN expects zero-centered target energies. Energies are shifted as:
$$E_{\text{referenced}} = E_{\text{DFT}} - \sum_i E_0(Z_i)$$
where:
- $E_0(\text{H}) = +1295.16198\text{ eV}$
- $E_0(\text{O}) = -4671.61107\text{ eV}$
- $E_0(\text{Si}) = -2659.59603\text{ eV}$

---

## 📂 Repository Architecture

The source code follows the three-layer architectural separation of **aims-PAX** (`src/aims_PAX/`):

```text
├── src/aims_PAX/                    # Modular 3-layer architecture package
│   ├── procedures/                  # High-level workflow orchestration
│   │   ├── preparation.py           # ALConfiguration, ALStateManager, ALEnsemble, ALMD
│   │   ├── active_learning.py       # ALProcedurePARSL (Closed-loop asynchronous active learning)
│   │   └── al_managers.py           # Decoupled managers (RunManager, DataManager, TrainManager)
│   └── tools/                       # Model interfaces & scientific utilities
│       ├── model_tools/
│       │   ├── painn_calculator.py  # PaiNN committee ASE calculator & uncertainty engine
│       │   ├── train_painn.py       # 1-epoch online fine-tuning (EMA) & 50-epoch post-AL convergence
│       │   └── mace_calculator.py   # MACE committee calculator wrapper
│       ├── uncertainty.py           # MolForceUncertainty & RollingAdaptiveThresholdManager
│       └── utilities/               # Trajectory dumps, MPI/Parsl runners, data I/O
├── al_5A_painn/                     # 3-System (5 Å gap) Closed-Loop PaiNN AL Pipeline
│   ├── run_painn_al_parsl.py        # Active learning entry point (ALProcedurePARSL)
│   ├── run_painn_al_parsl.slurm     # Slurm launch script for AL (1 GPU + 32 CPU cores)
│   ├── run_painn_converge.slurm     # Slurm script for 50-epoch post-AL final convergence
│   ├── al_dataset.xyz               # Accumulated labeled training dataset (535 frames)
│   ├── val.xyz                      # Validation dataset (145 frames)
│   ├── control.in                   # FHI-aims electronic structure parameters (PBE + vdW)
│   └── checkpoints/                 # Saved committee weights per cycle (cycle_0000 to cycle_0050)
├── al_5A_mace/                      # Reference aims-PAX MACE Active Learning Pipeline
│   ├── aimsprobe.yaml               # aims-PAX active learning settings
│   ├── model.yaml                   # MACE equivariant architecture settings
│   ├── run_mace_al_5A.slurm         # Slurm script for MACE active learning
│   ├── run_mace_converge.slurm      # Slurm script for MACE 200-epoch convergence
│   └── data/final/                  # Final MACE datasets (509 train, 142 val frames)
├── comparison_figures/              # Publication-quality comparison figures (Figures 1-6, PNG)
├── comparison_results/              # Detailed metrics, trajectory logs, and JSON summaries
├── generate_post_al_figures.py      # Post-AL figure generator (Figure 5 & Figure 6)
├── geometries/                      # Initial 5 Å, 10 Å, and 20 Å silica-water interface structures
├── archive/                         # Archived legacy scripts, monolithic prototypes & slurm logs
├── LICENSE                          # MIT License
├── .gitignore                       # Clean repository exclusions
└── README.md                        # Documentation
```

---

## 🚀 Getting Started

### Prerequisites
- **Python Environment**: Conda environment (`nff`) with `torch` (>= 2.5), `nff`, `ase`, `parsl`, `scipy`, `numpy`.
- **Quantum Chemistry Code**: `FHI-aims` compiled with MPI (Intel oneAPI or OpenMPI).
- **Species Defaults**: Standard FHI-aims species defaults directory (`defaults_2020/light`).

### Running PaiNN Active Learning
To execute the closed-loop active learning procedure across the 3 silica-water interfaces:
```bash
cd al_5A_painn
sbatch run_painn_al_parsl.slurm
```

### Running Post-AL Final Convergence
To train the committee for 50 full epochs on the accumulated active learning dataset:
```bash
cd al_5A_painn
sbatch run_painn_converge.slurm
```

### Generating Publication Figures
To regenerate Figures 5 and 6 comparing MACE and PaiNN (high-resolution PNG):
```bash
python generate_post_al_figures.py
```

---

## 📊 Outputs & Monitoring

During execution, the pipeline produces the following outputs in your working directory:

1. **Active Learning Dataset (`al_dataset.xyz`)**:
   - Contains all initial seed structures and newly labeled DFT frames.
   - Includes `REF_energy` (raw DFT eV), `REF_forces` (eV/Å), and atomic reference information.
2. **DFT Single-Point Archive (`dft_calculations/`)**:
   - Subdirectories formatted as `step_{step:06d}_traj_{traj_idx}/`.
   - Each contains `geometry.in`, full concatenated `control.in`, and complete `aims.out`.
3. **LAMMPS Trajectories (`trajectories/`)**:
   - `traj_traj{i}.lammpstrj`: Unwrapped coordinates recorded every 1,000 steps (`id type mol xu yu zu`).
   - `traj_stress_traj{i}.lammpstrj`: Unwrapped coordinates + diagonal virial stresses recorded every 5,000 steps (`id type mol xu yu zu c_stress[1] c_stress[2] c_stress[3]`).
4. **Model Checkpoints (`checkpoints/`)**:
   - Chronological directories: `cycle_{k}/model_{0..5}/best_model`.
5. **State File (`al_checkpoint.json`)**:
   - Stores current MD step counts, instantaneous positions, velocities, and threshold for seamless resumption.

---

## 🧪 Reproducibility Benchmark Suite

To prove mathematical and functional equivalence with [aims-PAX](https://github.com/tohenkes/aims-PAX/tree/main), we provide a 7-kernel reproducibility test suite in `benchmark_pipeline.py`.

### The 7 Verification Kernels

| # | Benchmark Kernel | Validation Target | Result |
|---|---|---|---|
| **1** | **Uncertainty Kernel Equivalence** | Compares PaiNN vector standard deviation against `aims_PAX.tools.uncertainty` | **$r = 1.000000$** (exact correlation) |
| **2** | **Reference Energy ($E_0$) Invariance** | Numerical reversibility of atomic baseline shifts: $E_{\text{ref}} = E - \sum E_0$ | **$0.00\text{ eV}$** roundtrip error |
| **3** | **Adaptive Threshold Dynamics** | State machine transitions and dynamic relaxation ($U_{\text{thresh}} \times 1.02$) | **Exact match** to analytical curve |
| **4** | **MD Checkpoint Restart Invariance** | Continuous MD vs. Stop-at-$N$, Checkpoint, Resume-to-$2N$ coordinates | **$< 10^{-10}\text{ Å}$** (zero drift) |
| **5** | **LAMMPS Custom Dump Compliance** | Header syntax, water clustering (`mol`), unwrapped coords, stress tensors | **100% compliant** (6 & 9 cols) |
| **6** | **Safe Loss & NaN/Inf Gradient Guard** | PaiNN autograd stability under compressed, unphysical configurations | **Intercepted cleanly** (zero grad) |
| **7** | **6-Model Committee Accuracy** | Energy and force evaluation against held-out DFT test set (`test.xyz`) | **$1.19\text{ meV/atom}$**, **$242\text{ meV/Å}$** |

### Running the Benchmark Suite on Cluster Partitions

In accordance with HPC cluster policies, never run intensive benchmarks or MD simulations on the login head node. Always submit to a compute partition via Slurm:

```bash
# Option A: Fast execution on compute partition (e.g. grafite, 2 CPUs)
sbatch run_benchmark_grafite.slurm

# Option B: Full GPU verification on GPU partition (etileno/metano, 1 GPU + 32 CPUs)
sbatch run_benchmark.slurm
```

---

## 🔄 Checkpointing & Resuming

If a job hits the 72-hour walltime limit or is canceled:
1. The pipeline catches the `SIGTERM` signal and saves the current positions, velocities, and cycle state to `al_checkpoint.json`.
2. To resume, simply resubmit the Slurm script:
   ```bash
   sbatch run_painn_al.slurm
   ```
3. The script automatically detects `al_checkpoint.json`, reloads the latest model weights from `checkpoints/`, restores trajectory coordinates, and continues MD without losing a step.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
