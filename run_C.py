from pathlib import Path
from shutil import copyfile
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent

# Author: Jimmy
def rebuild_pdbs(input_dir=ROOT / "data/Input", results_dir=ROOT / "results/C"):
    input_dir, results_dir = Path(input_dir), Path(results_dir)
    if input_dir.resolve() == results_dir.resolve():
        raise ValueError("Input and output directories must be different")
    results_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=results_dir) as tmp:
        for pdb in sorted(input_dir.iterdir()):
            if pdb.is_file() and pdb.suffix.lower() == ".pdb":
                copyfile(pdb, Path(tmp) / "input.pdb")
                subprocess.run([ROOT / "reference/pulchra", "-c", "input.pdb"], cwd=tmp, check=True)
                (Path(tmp) / "input.rebuilt.pdb").replace(results_dir / pdb.name)


if __name__ == "__main__":
    rebuild_pdbs(*sys.argv[1:])
