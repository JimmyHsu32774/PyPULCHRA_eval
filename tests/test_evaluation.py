"""AI-authored tests migrated from the evaluation notebook acceptance cases."""
from copy import deepcopy

import numpy as np
import pytest

from pypulchra.function import match_atom_keys, coordinate_rmsd, dihedral_degrees
from pypulchra.evaluate import (
    choose_oracle, classify_selection, evaluate_candidate_pool,
    evaluate_stages, evaluate_pools, _compare_structures,
)

def rmsd_stub(reference,prediction):
    return float(np.sqrt(np.mean(np.sum((np.asarray(reference)-np.asarray(prediction))**2,axis=1))))


def expect_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected ValueError')


def mock_ca():
    return dict(atom_keys=[('A',12,'','CA')],residue_names=['ALA'],coords=np.zeros((1,3)),segments=[[0]])


def mock_backbone():
    return dict(atom_keys=[('A',12,'',a) for a in ['N','CA','C','O']],residue_names=['ALA']*4,
                coords=np.array([[-1.,0.,0.],[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]]),segments=[[0,1,2,3]])


def mock_pool():
    return dict(residue_key=('A',12,''),residue_name='ALA',origin=[0.,0.,0.],scope='nearest_bin',
        candidates=[dict(id=i,atom_names=['CB'],local_coords=[[0.,y,0.]],global_coords=[[0.,y,0.]],
                         bin_id=[0,2,55],matching_score=0.,library_order=0)
                    for i,y in [('a',1.),('b',-1.)]])


def mock_selected():
    bb=mock_backbone()
    return dict(atom_keys=bb['atom_keys']+[('A',12,'','CB')],residue_names=['ALA']*5,
                coords=np.vstack([bb['coords'],[0.,-1.,0.]]),segments=[[0,1,2,3,4]],
                selected_candidate_ids=['b'])


def test_notebook_y1():
    reference = [('A', 1, '', 'N'), ('A', 1, '', 'CA'), ('A', 1, '', 'C')]
    prediction = [reference[2], reference[0]]
    assert match_atom_keys(reference, prediction) == ([0, 2], [1, 0])
    expect_value_error(lambda: match_atom_keys([reference[0]] * 2, prediction))


def test_notebook_y2():
    reference = np.array([[0., 0., 0.], [2., 0., 0.]])
    prediction = np.array([[0., 0., 0.], [2., 2., 0.]])
    assert np.isclose(coordinate_rmsd(reference, prediction), np.sqrt(2))
    expect_value_error(lambda: coordinate_rmsd(np.empty((0, 3)), np.empty((0, 3))))


def test_notebook_y3():
    points = [[0., 1., 0.], [0., 0., 0.], [1., 0., 0.], [1., 0., 1.]]
    assert np.isclose(dihedral_degrees(*points), 90.)
    expect_value_error(lambda: dihedral_degrees(points[0], points[1], points[1], points[3]))


def test_notebook_j2():
    ids, errors = [10, 11, 12], [1.8, 0.6, 0.6]
    assert choose_oracle(ids, errors) == (11, 0.6)
    assert choose_oracle([10], [np.nan]) == (None, None)
    expect_value_error(lambda: choose_oracle([10], [-0.1]))


def test_notebook_j3():
    errors, threshold = [2., 0.4], 1.
    assert classify_selection(2., errors, threshold) == 'recoverable_selection_error'
    assert classify_selection(0.4, errors, threshold) == 'selected_correct'
    assert classify_selection(2., [2., 1.5], threshold) == 'uncovered'
    expect_value_error(lambda: classify_selection(3., errors, threshold))


def test_notebook_i2():
    reference, ids = [[0.,0.,0.]], ['a','b']
    coords = [[[2.,0.,0.]], [[0.,0.,0.]]]
    result = evaluate_candidate_pool(reference, ids, coords, 'a', rmsd_stub, 1.)
    assert result['selected_id']=='a' and result['oracle_id']=='b'
    assert result['selected_error']==2. and result['oracle_error']==0.
    assert result['label']=='recoverable_selection_error' and result['n_candidates']==2
    expect_value_error(lambda: evaluate_candidate_pool(reference, ids, coords, 'missing', rmsd_stub, 1.))


def test_boundaries_y1():
    assert match_atom_keys([], []) == ([], [])
    assert match_atom_keys([('A',1,'','CA')], [('B',1,'','CA')]) == ([], [])
    assert match_atom_keys([('A',1,'A','CA')], [('A',1,'','CA')]) == ([], [])
    expect_value_error(lambda: match_atom_keys([('A',1,'','CA')], [('A',1,'','CA')]*2))
    print('Y1 additional checks PASS: empty inputs, chain IDs, insertion codes, duplicate prediction')


def test_boundaries_y2():
    coords = np.array([[0.,0.,0.],[1.,2.,3.]])
    original = coords.copy()
    assert coordinate_rmsd(coords, coords) == 0.
    assert np.isclose(coordinate_rmsd(coords, coords+[1.,2.,2.]),3.)
    assert np.isclose(coordinate_rmsd([[0.,0.,0.]],[[1e200,0.,0.]]),1e200)
    for bad in ([], [[0.,0.]], [[np.nan,0.,0.]], [[np.inf,0.,0.]]):
        expect_value_error(lambda: coordinate_rmsd(coords,bad))
    np.testing.assert_array_equal(coords,original)
    print('Y2 additional checks PASS: zero RMSD, translation, large values, invalid input, input preservation')


def test_boundaries_y3():
    assert np.isclose(dihedral_degrees([0.,1.,0.],[0.,0.,0.],[1.,0.,0.],[1.,0.,-1.]),-90.)
    assert np.isclose(dihedral_degrees([0.,1.,0.],[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]),0.)
    assert dihedral_degrees([0.,1.,0.],[0.,0.,0.],[1.,0.,0.],[1.,-1.,0.]) == -180.
    expect_value_error(lambda: dihedral_degrees([0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]))
    expect_value_error(lambda: dihedral_degrees([np.nan,0.,0.],[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]))
    print('Y3 additional checks PASS: negative, coplanar, -180 boundary, degenerate and nonfinite geometry')


def test_boundaries_j2():
    assert choose_oracle([],[]) == (None,None)
    assert choose_oracle(['a','b','c'],[np.nan,0.2,0.2]) == ('b',0.2)
    assert choose_oracle([np.int64(7)],[0.]) == (7,0.)
    for ids, errors in [(['a'],[1.,2.]),(['a','a'],[1.,2.]),([''],[1.]),([True],[1.]),(['a'],[np.inf]),(['a'],[-1.])]:
        expect_value_error(lambda: choose_oracle(ids,errors))
    print('J2 additional checks PASS: empty pool, NaN, ties, integer IDs and invalid inputs')


def test_boundaries_j3():
    assert classify_selection(1.,[1.,2.],1.) == 'selected_correct'
    assert classify_selection(2.,[2.,1.],1.) == 'recoverable_selection_error'
    for selected, errors, threshold in [(2.,[],1.),(2.,[2.,np.nan],1.),(2.,[2.],-1.),(3.,[2.],1.)]:
        expect_value_error(lambda: classify_selection(selected,errors,threshold))
    expect_value_error(lambda: classify_selection(1000000.1,[1000000.],1.))
    print('J3 additional checks PASS: threshold equality, invalid errors, missing selection, absolute tolerance')


def test_boundaries_i2():
    reference = np.array([[0.,0.,0.],[1.,0.,0.]])
    candidates = np.array([reference+[0.,2.,0.],reference])
    saved = candidates.copy()
    row = evaluate_candidate_pool(reference,['a','b'],candidates,'a',coordinate_rmsd,1.)
    assert row['oracle_error'] <= row['selected_error']
    assert row['selected_error']==2. and row['oracle_error']==0.
    tied = evaluate_candidate_pool(reference,['first','second'],[reference,reference],'second',coordinate_rmsd,0.)
    assert tied['oracle_id']=='first' and tied['label']=='selected_correct'
    for ids, xyz, chosen in [(['a','a'],candidates,'a'),(['a','b'],candidates,'missing'),(['a'],candidates,'a')]:
        expect_value_error(lambda: evaluate_candidate_pool(reference,ids,xyz,chosen,coordinate_rmsd,1.))
    for metric in (lambda r,p: np.nan, lambda r,p: -1., lambda r,p: [0.]):
        expect_value_error(lambda: evaluate_candidate_pool(reference,['a','b'],candidates,'a',metric,1.))
    np.testing.assert_array_equal(candidates,saved)
    print('I2 additional checks PASS: fixed frame, tied Oracle, candidate IDs, callback validation, input preservation')


def test_stage_and_pool_evaluation():
    native = mock_selected()
    native['coords'][-1] += [0.,2.,0.]
    selected = mock_selected()
    python_result = {'selected':selected, 'corrected':deepcopy(selected),
                     'candidate_pools':[dict(mock_pool(),selected_id='b')]}
    config = {'protein_id':'MOCK','chain':'A','candidate_scope':'nearest_bin','threshold':1.}
    c_result = {'status':'ok','structure':deepcopy(selected)}
    metrics = evaluate_stages(native,python_result,c_result,config)
    assert np.isclose(metrics['c_vs_native_backbone_rmsd_angstrom'],0.,atol=1e-12)
    assert np.isclose(metrics['c_vs_native_sidechain_rmsd_angstrom'],2.)
    assert metrics['python_vs_c_rmsd']==0.
    rows = evaluate_pools(native,python_result,config)
    assert rows[0]['oracle_id']=='a' and rows[0]['selected_id']=='b'
    assert rows[0]['label']=='recoverable_selection_error'
    assert np.isclose(rows[0]['selected_error'],2.) and np.isclose(rows[0]['oracle_error'],0.,atol=1e-12)

    # One backbone fit must also remove a rigid translation from the side-chain comparison.
    shifted = deepcopy(selected)
    shifted['coords'] += [5.,-3.,2.]
    matched, _, _ = _compare_structures(selected,shifted,align=True)
    assert matched['backbone_rmsd'] < 1e-12 and matched['sidechain_rmsd'] < 1e-12
    direct, _, _ = _compare_structures(selected,shifted,align=False)
    assert np.isclose(direct['all_heavy_rmsd'],np.sqrt(38))

    # Explicit chain mapping, atom order and exclusions.
    blank = deepcopy(selected)
    blank['atom_keys'] = [('',*key[1:]) for key in blank['atom_keys']]
    mapped = evaluate_stages(native,python_result,{'status':'ok','structure':blank},
                             dict(config,c_chain_map={'':'A'}))
    assert mapped['n_matched_backbone_atoms']==4 and mapped['n_matched_sidechain_atoms']==1
    reordered = deepcopy(selected)
    reordered['atom_keys'] = reordered['atom_keys'][::-1]
    reordered['coords'] = reordered['coords'][::-1]
    assert _compare_structures(selected,reordered,align=False)[0]['all_heavy_rmsd']==0.

    missing = deepcopy(native)
    missing['atom_keys'] = missing['atom_keys'][:-1]
    missing['coords'] = missing['coords'][:-1]
    excluded = evaluate_pools(missing,python_result,config)
    assert excluded[0]['status']=='excluded' and excluded[0]['reason']=='missing_native_atoms'
    assert excluded[0]['label'] is None

    bad = deepcopy(python_result)
    bad['candidate_pools'][0]['selected_id'] = 'a'
    expect_value_error(lambda: evaluate_pools(native,bad,config))
    bad = deepcopy(python_result)
    bad['candidate_pools'][0]['candidates'][1]['id'] = 'a'
    expect_value_error(lambda: evaluate_pools(native,bad,config))
    duplicate = deepcopy(selected)
    duplicate['atom_keys'][-1] = duplicate['atom_keys'][0]
    expect_value_error(lambda: _compare_structures(native,duplicate,align=False))
    print('I3 evaluation checks PASS: shared fit, direct parity, chain mapping, order, exclusions and candidate identity')
