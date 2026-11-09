import json
import os
from pathlib import Path
import numpy as np
from pypulchra.library import parse_backbone_library
from pypulchra.reconstruct import rebuild_backbone

ROOT = Path(__file__).resolve().parents[1]

# Author: Yitian
def test_y7_notebook_cases_and_c_reference():
    # Y7 tests: the public example plus terminal, break, PRO and fixture checks.
    empty = rebuild_backbone({'atom_keys': [], 'residue_names': [],
                              'coords': np.empty((0, 3)), 'segments': []}, {}, {})
    assert empty['atom_keys'] == []
    assert np.asarray(empty['coords']).shape == (0, 3)
    assert empty['diagnostics'] == []
    
    # Author: Yitian
    def ca_record(keys, names, coords, segments=None):
        return {'atom_keys': list(keys),
                'residue_names': list(names),
                'coords': np.asarray(coords, dtype=float),
                'segments': segments if segments is not None else [list(range(len(keys)))]}
    
    
    # Author: Yitian
    def ideal_helix(n, radius=2.3, rise=1.5, turn=100.0):
        angle = np.radians(turn) * np.arange(n)
        return np.column_stack((radius * np.cos(angle),
                                radius * np.sin(angle),
                                rise * np.arange(n)))
    
    
    library_path = str(ROOT / 'reference/nco_data.h')
    assert library_path is not None, 'put the pinned PULCHRA source in projects/PULCHRA/src'
    library = parse_backbone_library(library_path)
    print('library:', library_path)
    print('fragments:', len(library['standard']), 'standard,', len(library['proline']), 'proline')
    
    # one short chain: N/CA/C/O for every residue plus a C-terminal OXT
    n = 8
    keys = [('A', residue + 1, '', 'CA') for residue in range(n)]
    helix = rebuild_backbone(ca_record(keys, ['ALA'] * n, ideal_helix(n)), library, {})
    assert len(helix['atom_keys']) == 4 * n + 1
    assert helix['atom_keys'][0] == ('A', 1, '', 'N')
    assert helix['atom_keys'][-1] == ('A', n, '', 'OXT')
    assert [d['status'] for d in helix['diagnostics']] == ['rebuilt'] * (n + 1)
    
    placed = dict(zip(helix['atom_keys'], helix['coords']))
    for residue in range(1, n + 1):
        ca = placed[('A', residue, '', 'CA')]
        assert 1.2 < np.linalg.norm(placed[('A', residue, '', 'N')] - ca) < 1.7
        assert 1.2 < np.linalg.norm(placed[('A', residue, '', 'C')] - ca) < 1.7
        assert 1.1 < np.linalg.norm(placed[('A', residue, '', 'O')]
                                    - placed[('A', residue, '', 'C')]) < 1.4
    
    # a chain break must not leak a four-CA window across the gap
    left = ideal_helix(6)
    right = ideal_helix(6) + np.array([0.0, 0.0, 40.0])
    split_keys = [('A', residue + 1, '', 'CA') for residue in range(6)]
    split_keys += [('B', residue + 1, '', 'CA') for residue in range(6)]
    joined = rebuild_backbone(ca_record(split_keys, ['ALA'] * 12, np.vstack((left, right)),
                                        [list(range(6)), list(range(6, 12))]), library, {})
    alone = rebuild_backbone(ca_record(split_keys[6:], ['ALA'] * 6, right), library, {})
    assert len(joined['atom_keys']) == 2 * (4 * 6 + 1)
    assert len(joined['segments']) == 2
    joined_pos = dict(zip(joined['atom_keys'], joined['coords']))
    for key, xyz in zip(alone['atom_keys'], alone['coords']):
        assert np.allclose(joined_pos[key], xyz)
    
    # proline: the window that follows a PRO residue uses nco_stat_pro
    names = ['ALA', 'PRO'] + ['ALA'] * (n - 2)
    pro = rebuild_backbone(ca_record(keys, names, ideal_helix(n)), library, {})
    assert [d['library'] for d in pro['diagnostics']].count('proline') == 1
    assert pro['diagnostics'][2]['library'] == 'proline'
    
    # invalid segments, a missing library and unknown options are rejected
    for call in (lambda: rebuild_backbone(ca_record(keys, names, ideal_helix(n),
                                                    [[0, 1, 2], [3, 4, 5, 6]]), library, {}),
                 lambda: rebuild_backbone(ca_record(keys, names, ideal_helix(n)), {}, {}),
                 lambda: rebuild_backbone(ca_record(keys, names, ideal_helix(n)), library,
                                          {'fix_the_rest': True})):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError('expected ValueError')
    
    # a segment shorter than five CA is reported instead of inventing padding
    short = rebuild_backbone(ca_record(keys[:4], ['ALA'] * 4, ideal_helix(4)), library, {})
    assert short['atom_keys'] == [] and short['coords'].shape == (0, 3)
    assert len(short['diagnostics']) == 4
    assert all(d['status'] == 'excluded' for d in short['diagnostics'])
    
    # the pinned C reference: same input, original frame, no alignment
    fixture_dir = str(ROOT / 'tests/fixtures')
    fixture_path = os.path.normpath(os.path.join(fixture_dir, 'y7_1ubq_c_backbone.json'))
    assert os.path.isfile(fixture_path), 'required C fixture missing'
    if True:
        fixture = json.load(open(fixture_path, encoding='utf-8'))
        ca = fixture['input_ca']
        result = rebuild_backbone(ca_record(ca['atom_keys'], ca['residue_names'], ca['coords']),
                                  library, {'source_commit': fixture['source']['commit']})
    
        reference = dict(zip(map(tuple, fixture['reference']['atom_keys']),
                             np.array(fixture['reference']['coords'])))
        prediction = dict(zip(result['atom_keys'], result['coords']))
        assert set(prediction) == set(reference)
        displacement = np.array([np.linalg.norm(prediction[key] - reference[key])
                                 for key in reference])
        print('1UBQ vs C reference: %d atoms, max %.4f A, rms %.4f A (tolerance %.2f A)'
              % (len(displacement), displacement.max(),
                 np.sqrt(np.mean(displacement ** 2)),
                 fixture['tolerance']['atol_angstrom']))
        assert displacement.max() < fixture['tolerance']['atol_angstrom']
    
        native = dict(zip(map(tuple, fixture['native_backbone']['atom_keys']),
                          np.array(fixture['native_backbone']['coords'])))
        error = np.array([np.linalg.norm(xyz - native[key]) for key, xyz in prediction.items()])
        print('1UBQ vs experimental structure: rms %.3f A, max %.3f A'
              % (np.sqrt(np.mean(error ** 2)), error.max()))
