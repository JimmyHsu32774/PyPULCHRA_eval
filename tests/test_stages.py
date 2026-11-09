import numpy as np
import pytest


# Test written by Codex.
def test_i1_owns_snapshots_and_rejects_bad_selection():
    from pypulchra.reconstruct import run_reconstruction
    calls = []
    original = {'coords': np.zeros((1, 3))}

    def optimize(ca, options):
        calls.append('optimize')
        ca['coords'][0, 0] = 1
        options['changed'] = True
        return ca

    def backbone(ca, options):
        calls.append('backbone')
        assert 'changed' not in options
        ca['coords'][0, 0] = 2
        return ca

    def candidates(ca, backbone, options):
        calls.append('candidates')
        assert ca['coords'][0, 0] == 1
        return [{'candidate_ids': ['a', 'b'], 'coordinates': [[[1, 0, 0]], [[2, 0, 0]]]}]

    def select(backbone, pools, options):
        calls.append('select')
        pools[0]['coordinates'][0][0][0] = 99
        return {'selected_candidate_ids': ['b'], 'coords': np.ones((1, 3))}

    def correct(selected, pools, options):
        calls.append('correct')
        assert pools[0]['selected_id'] == 'b'
        selected['coords'][0, 0] = 3
        return selected

    stages = dict(optimize_ca=optimize, backbone=backbone, candidates=candidates, select=select, correct=correct)
    result = run_reconstruction(original, {}, stages)
    assert calls == ['optimize', 'backbone', 'candidates', 'select', 'correct']
    assert original['coords'][0, 0] == 0
    assert result['ca_optimized']['coords'][0, 0] == 1
    assert result['selected']['coords'][0, 0] == 1
    assert result['candidate_pools_before_selection'][0]['coordinates'][0][0][0] == 1
    calls.clear()
    with pytest.raises(ValueError):
        run_reconstruction(original, {}, {**stages, 'correct': None})
    assert calls == []
    stages['select'] = lambda *args: {'selected_candidate_ids': ['missing']}
    with pytest.raises(ValueError, match='outside the pool'):
        run_reconstruction(original, {}, stages)


# Test written by Codex.
def test_backbone_example(tmp_path):
    from run import ROOT, run_backbone
    from pypulchra.function import parse_pdb_atoms

    source = ROOT / 'data/demo/Input/y7_1ubq_ca.pdb'
    sample = tmp_path / 'eight_ca.pdb'
    sample.write_text(''.join(source.read_text().splitlines(keepends=True)[:8]))
    output = tmp_path / 'rebuilt.pdb'
    result = run_backbone(sample, output, optimize_ca=True)
    original = np.asarray(result['ca_input']['coords'])
    optimized = result['ca_optimized']
    assert optimized['stage_status'] == 'executed'
    assert optimized['diagnostics'][0]['iterations'] > 0
    assert not np.allclose(original, optimized['coords'])
    backbone = result['backbone']
    ca_rows = [i for i, key in enumerate(backbone['atom_keys']) if key[3] == 'CA']
    np.testing.assert_array_equal(backbone['coords'][ca_rows], optimized['coords'])
    assert len(backbone['atom_keys']) == 33 and backbone['atom_keys'][-1][3] == 'OXT'
    atoms = parse_pdb_atoms(output.read_text().splitlines())
    assert [atom['key'] for atom in atoms] == backbone['atom_keys']
    np.testing.assert_allclose([atom['xyz'] for atom in atoms], backbone['coords'], atol=0.0005, rtol=0)
    assert result['corrected']['stage_status'] == 'disabled'


# Test written by Codex.
def test_retrieve_two_conformers_and_place_in_a_rotated_frame():
    from pypulchra.function import make_local_frame
    from pypulchra.reconstruct import retrieve_rotamers, place_candidate

    library = {
        'rot_stat_idx': [['ALA', 0, 2, 55, 2, 0]],
        'rot_stat_coords': [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]],
        'heavy_atoms': {'ALA': ['CB']},
    }
    candidates = retrieve_rotamers('ALA', [0, 2, 55], library, 'nearest_bin')
    assert [row[0] for row in candidates] == ['ALA:0:0', 'ALA:0:1']
    assert candidates[1][1:4] == [[0, 2, 55], 0.0, ['CA', 'CB']]
    local = candidates[1][4]
    assert local == [[0, 0, 0], [0, 1, 0]]
    origin, axes = make_local_frame([1, 2, 3], [3, 4, 5], [2, 2, 1])
    np.testing.assert_allclose(place_candidate(local, origin, axes), np.array(local) @ axes.T + origin)


# Test written by Codex.
def test_clash_helpers():
    from pypulchra.clashes import build_excluded_pairs, find_close_pairs

    structure = {'atom_keys': [('A', 1, '', 'C'), ('A', 2, '', 'N'), ('B', 1, '', 'CA')],
                 'coords': [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]}
    excluded = build_excluded_pairs(structure)
    assert excluded == {(0, 1)}
    np.testing.assert_array_equal(find_close_pairs(structure['coords'], 2, excluded), [[0, 2]])
