import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pypulchra.function import read_ca_record

ROOT = Path(__file__).resolve().parents[1]


def test_missing_pulchra(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    result = subprocess.run([sys.executable, str(ROOT / "run_C.py")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "--pulchra PATH" in result.stderr


@pytest.mark.skipif(shutil.which("pulchra") is None, reason="Install pulchra to run the C integration tests")
@pytest.mark.parametrize("optimize", [False, True])
def test_c_runner_ca_coordinates(tmp_path, optimize):
    source = ROOT / "data/demo/Input/y7_1ubq_ca.pdb"
    # Use the installed C program directly as the reference for this build.
    shutil.copyfile(source, tmp_path / "input.pdb")
    flags = [] if optimize else ["-c"]
    direct = subprocess.run([shutil.which("pulchra"), *flags, "input.pdb"], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    output = tmp_path / "runner output"
    command = [sys.executable, str(ROOT / "run_C.py"), str(source.parent), str(output)]
    if optimize:
        command.append("--optimize-ca")
        command.extend(["--pulchra", shutil.which("pulchra")])
    result = subprocess.run(command, capture_output=True, text=True)
    if optimize and "Initial coordinates will be preserved" in direct.stdout:
        assert result.returncode != 0
        assert "_PRESERVE = 0" in result.stderr
        assert not (output / source.name).exists()
        return

    result.check_returncode()
    original = read_ca_record(source)
    rebuilt = read_ca_record(output / source.name)
    expected = read_ca_record(tmp_path / "input.rebuilt.pdb")
    # PULCHRA writes a blank chain ID for the single-chain demo.
    assert [key[1:] for key in rebuilt["atom_keys"]] == [key[1:] for key in original["atom_keys"]]
    assert rebuilt["residue_names"] == original["residue_names"]
    np.testing.assert_array_equal(rebuilt["coords"], expected["coords"])
    if optimize:
        assert not np.array_equal(rebuilt["coords"], original["coords"])
    else:
        np.testing.assert_array_equal(rebuilt["coords"], original["coords"])
