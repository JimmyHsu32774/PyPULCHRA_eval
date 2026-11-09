# PyPULCHRA

Rebuild a protein backbone from C-alpha coordinates. Run these commands from the `PyPULCHRA_integrate` directory.

Create the environment once. On your next run, just activate it.

```bash
conda create -n pypulchra python=3.12 numpy scipy pytest
conda activate pypulchra
```

### Build PULCHRA for the C runner

The Python reconstruction can run on its own. To run the C comparison, compile PULCHRA once for your platform. With a Unix-style C compiler, run these commands from a separate directory.

```bash
git clone https://github.com/euplotes/pulchra.git
cd pulchra
cc -O3 -o pulchra pulchra.c pulchra_data.c -lm
```

Put the directory containing `pulchra` (`pulchra.exe` on Windows) on your `PATH`, or pass the executable's path to the runner from the PyPULCHRA project directory.

```bash
python run_C.py data/demo/Input data/demo/output/C --pulchra "/path/to/pulchra"
```

**To enable Cα optimization**, check the default in `pulchra.c` before compiling. In upstream revision `23a3e4a`, change `int _PRESERVE = 1;` to `int _PRESERVE = 0;` and run the same compiler command. This enables both modes in one executable. The runner passes `-c` to keep the input Cα positions and omits it for `--optimize-ca`. If the executable reports that it preserves input coordinates during an optimization run, the runner stops before writing that result. [Upstream optimizer condition](https://github.com/euplotes/pulchra/blob/23a3e4afc01ab48fe81e5d4a8d5902352d8f0977/pulchra.c#L3636)

### Run the 1UBQ demo

These commands keep the input C-alpha coordinates.

```bash
python run.py reconstruct --input data/demo/Input/y7_1ubq_ca.pdb --output data/demo/output/1ubq_backbone.pdb
python run_C.py data/demo/Input data/demo/output/C
```

With an optimization-enabled C build, add `--optimize-ca` to both commands to optimize the same trace before reconstruction.

```bash
python run.py reconstruct --input data/demo/Input/y7_1ubq_ca.pdb --output data/demo/output/1ubq_backbone_optimized.pdb --optimize-ca
python run_C.py data/demo/Input data/demo/output/C_optimized --optimize-ca
```

Compare the CA atoms in `1ubq_backbone_optimized.pdb` and `C_optimized/y7_1ubq_ca.pdb` under `data/demo/output/`. The C runner keeps the input filename and also rebuilds side chains. For 1UBQ, its blank chain field corresponds to chain A.

### Use another PDB

```bash
python run.py reconstruct --input data/Input/108L_A.pdb --output results/108L_A.pdb
```

### Run a whole folder

To compare optimized results for a whole folder, use the same input directory for both runs. Each output keeps its input filename.

```bash
for pdb in data/Input/*.pdb; do
  python run.py reconstruct --input "$pdb" --output "results/Python_optimized/$(basename "$pdb")" --optimize-ca
done
python run_C.py data/Input results/C_optimized --optimize-ca
```

For a run with optimization off, omit `--optimize-ca` from both commands and use `results/Python/` and `results/C/` as the output directories.

Run the tests with `python -m pytest -q tests`.
