# PyPULCHRA

Rebuild a protein backbone from C-alpha coordinates. Run these commands from the `PyPULCHRA_integrate` directory.

Create the environment once. On your next run, just activate it.

```bash
conda create -n pypulchra python=3.12 numpy scipy pytest
conda activate pypulchra
```

### Run the 1UBQ demo

```bash
python run.py reconstruct --input data/demo/Input/y7_1ubq_ca.pdb --output data/demo/output/1ubq_backbone.pdb
```

To optimize the C-alpha trace first, add `--optimize-ca` and write to a different file.

```bash
python run.py reconstruct --input data/demo/Input/y7_1ubq_ca.pdb --output data/demo/output/1ubq_backbone_optimized.pdb --optimize-ca
```

The original C program can process the same demo. Its runner takes two directories, preserves the input C-alpha coordinates and keeps the input filename.

```bash
python run_C.py data/demo/Input data/demo/output/C
```

### Use another PDB

```bash
python run.py reconstruct --input data/Input/108L_A.pdb --output results/108L_A.pdb
```

### Run a whole folder

To rebuild every `.pdb` file directly inside `data/Input/` with Python, run this loop. Each output keeps its input filename under `results/Python/`.

```bash
for pdb in data/Input/*.pdb; do
  python run.py reconstruct --input "$pdb" --output "results/Python/$(basename "$pdb")"
done
```

To optimize every C-alpha trace first, add `--optimize-ca` to the Python command inside the loop.

The C runner reads the whole folder and writes each rebuilt PDB to `results/C/`.

```bash
python run_C.py data/Input results/C
```

Run the tests with `python -m pytest -q tests`.
