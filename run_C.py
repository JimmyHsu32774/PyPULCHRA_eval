import argparse
import subprocess
from pathlib import Path
from shutil import copyfile, which
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent


# Written by Codex.
def rebuild_pdbs(input_dir=ROOT / "data/Input", results_dir=None, optimize_ca=False, pulchra="pulchra"):
    if results_dir is None:
        results_dir = ROOT / "results" / ("C_optimized" if optimize_ca else "C")
    input_dir, results_dir = Path(input_dir), Path(results_dir)
    if input_dir.resolve() == results_dir.resolve():
        raise ValueError("Input and output directories must be different")
    executable = which(pulchra)
    if executable is None:
        raise FileNotFoundError(
            f"Cannot find PULCHRA executable '{pulchra}'. Add it to PATH or pass --pulchra PATH."
        )
    command = [str(Path(executable).resolve())]
    if not optimize_ca:
        command.append("-c")
    command.append("input.pdb")

    results_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=results_dir) as tmp:
        for pdb in sorted(input_dir.iterdir()):
            if pdb.is_file() and pdb.suffix.lower() == ".pdb":
                copyfile(pdb, Path(tmp) / "input.pdb")
                result = subprocess.run(command, cwd=tmp, check=True,
                                        stdout=subprocess.PIPE if optimize_ca else None, text=True)
                if optimize_ca and "Initial coordinates will be preserved" in result.stdout:
                    raise ValueError(
                        "PULCHRA reported that it preserves the input coordinates. "
                        "For --optimize-ca, use a build with _PRESERVE = 0 (see README.md)."
                    )
                (Path(tmp) / "input.rebuilt.pdb").replace(results_dir / pdb.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild a folder of PDB files with PULCHRA.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=ROOT / "data/Input")
    parser.add_argument("results_dir", nargs="?", type=Path, help="Defaults to results/C or results/C_optimized")
    parser.add_argument("--pulchra", default="pulchra", help="PULCHRA executable (default: pulchra on PATH)")
    parser.add_argument("--optimize-ca", action="store_true", help="Use the installed build's C-alpha optimizer")
    args = parser.parse_args()
    try:
        rebuild_pdbs(args.input_dir, args.results_dir, optimize_ca=args.optimize_ca, pulchra=args.pulchra)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
