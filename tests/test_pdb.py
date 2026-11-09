import numpy as np
import pytest
from pypulchra.function import parse_pdb_atoms

LINE = 'ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C  '

# Test written by Codex.
def test_example_and_empty():
    assert parse_pdb_atoms([LINE]) == [{'key': ('A', 1, '', 'CA'), 'xyz': (1., 2., 3.)}]
    assert parse_pdb_atoms(['HEADER example']) == []

# Test written by Codex.
def test_insertion_negative_residue_and_altloc():
    line = LINE[:22] + '  -1' + 'B' + LINE[27:]
    assert parse_pdb_atoms([line])[0]['key'] == ('A', -1, 'B', 'CA')
    assert parse_pdb_atoms([LINE[:16] + 'B' + LINE[17:]]) == []
    assert len(parse_pdb_atoms([LINE[:16] + 'A' + LINE[17:]])) == 1

# Test written by Codex.
def test_first_model_only_and_hydrogen_retained():
    assert len(parse_pdb_atoms([LINE, 'ENDMDL', LINE])) == 1
    hydrogen = LINE[:12] + ' H  ' + LINE[16:]
    assert parse_pdb_atoms([hydrogen])[0]['key'][-1] == 'H'
    assert parse_pdb_atoms(['HETATM' + LINE[6:]]) == []

# Test written by Codex.
@pytest.mark.parametrize('lines', [[LINE, LINE], [LINE[:30] + '     nan' + LINE[38:]], [LINE[:30] + ' invalid' + LINE[38:]], [LINE[:40]], [42]])
def test_invalid_records(lines):
    with pytest.raises(ValueError):
        parse_pdb_atoms(lines)
