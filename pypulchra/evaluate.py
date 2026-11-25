"""Candidate selection and structure evaluation helpers."""

from numbers import Integral

import numpy as np

from .function import coordinate_rmsd, fit_rigid_transform, match_atom_keys


# Task: J2
# Author: Yitian Xiao (J2)
def choose_oracle(candidate_ids, errors):
    """Return the (id, error) pair with the smallest error; the first one wins."""
    ids = list(candidate_ids)
    errors = np.asarray(errors, dtype=float)

    if errors.ndim != 1 or len(errors) != len(ids):
        raise ValueError('need exactly one error per candidate')
    for cid in ids:
        if isinstance(cid, bool) or not isinstance(cid, (str, Integral)) or cid == '':
            raise ValueError('candidate IDs must be nonempty strings or integers')
    if len(set(ids)) != len(ids):
        raise ValueError('candidate IDs must be unique')
    if (errors < 0).any() or np.isinf(errors).any():
        raise ValueError('errors must be nonnegative and finite')

    best_row = None
    for row in range(len(errors)):
        if np.isnan(errors[row]):
            continue
        if best_row is None or errors[row] < errors[best_row]:
            best_row = row
    if best_row is None:
        return None, None
    return ids[best_row], float(errors[best_row])


# Task: J3
# Author: Yitian Xiao (J3)
def classify_selection(selected_error, candidate_errors, threshold):
    """Label the original selection relative to one error threshold."""
    errors = np.asarray(candidate_errors, dtype=float)
    if errors.ndim != 1 or len(errors) == 0:
        raise ValueError('candidate errors must be a nonempty 1D sequence')
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError('candidate errors must be finite and nonnegative')

    if not np.isfinite(selected_error) or not np.isfinite(threshold):
        raise ValueError('selected error and threshold must be finite')
    if selected_error < 0 or threshold < 0:
        raise ValueError('selected error and threshold must be nonnegative')
    if not np.isclose(errors, selected_error, atol=1e-10, rtol=0).any():
        raise ValueError('selected error is not in the candidate pool')

    if selected_error <= threshold:
        return 'selected_correct'
    if errors.min() <= threshold:
        return 'recoverable_selection_error'
    return 'uncovered'


# Task: C4
def compare_stage_metrics(before, after, atol=1e-10):
    pass


# Task: T2
def summarize_recovery(labels):
    pass


# Task: T3
def paired_metric_summary(before, after, atol=1e-10):
    pass


# Task: T5
def evaluate_chi1(reference_atoms, predicted_atoms, definitions, angle_fn, error_fn, tolerance=40.0):
    pass


# AI-assisted evaluation infrastructure from the notebook reference module.
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}
ALL_BACKBONE_ATOMS = BACKBONE_ATOMS | {'OXT'}


def _prepare_structure(structure, chain_map=None):
    """Collect atom keys and coordinates, renaming chains where asked."""
    keys = [tuple(key) for key in structure['atom_keys']]
    coords = np.asarray(structure['coords'], dtype=float)
    if len(keys) == 0:
        coords = coords.reshape(0, 3)
    if coords.shape != (len(keys), 3):
        raise ValueError('atom keys and coordinates do not match')
    if not np.isfinite(coords).all():
        raise ValueError('coordinates must be finite')
    for key in keys:
        if len(key) != 4 or not isinstance(key[3], str) or key[3] == '':
            raise ValueError('atom keys must be (chain, number, insertion, name)')

    mapping = chain_map or {}
    renamed = [(mapping.get(key[0], key[0]), key[1], key[2], key[3]) for key in keys]
    if len(set(renamed)) != len(renamed):
        raise ValueError('duplicate atom keys after chain mapping')
    return renamed, coords


def _heavy_atom(key):
    return not key[3].lstrip('0123456789').startswith(('H', 'D'))


def _compare_structures(reference, prediction, *, align, chain_map=None):
    """Compare heavy atoms; every mask shares one backbone superposition."""
    ref_keys, ref_xyz = _prepare_structure(reference)
    pred_keys, pred_xyz = _prepare_structure(prediction, chain_map)

    ref_rows, pred_rows = match_atom_keys(ref_keys, pred_keys)
    keys, ref, pred = [], [], []
    for i, j in zip(ref_rows, pred_rows):
        if _heavy_atom(ref_keys[i]):
            keys.append(ref_keys[i])
            ref.append(ref_xyz[i])
            pred.append(pred_xyz[j])
    ref = np.asarray(ref, dtype=float).reshape(-1, 3)
    pred = np.asarray(pred, dtype=float).reshape(-1, 3)

    backbone = [key[3] in BACKBONE_ATOMS for key in keys]
    sidechain = [key[3] not in ALL_BACKBONE_ATOMS for key in keys]
    bb_rows = [row for row, flag in enumerate(backbone) if flag]
    sc_rows = [row for row, flag in enumerate(sidechain) if flag]
    ref_heavy = {key for key in ref_keys if _heavy_atom(key)}
    pred_heavy = {key for key in pred_keys if _heavy_atom(key)}

    result = dict(status='ok',
                  n_matched_all=len(keys),
                  n_matched_backbone_atoms=len(bb_rows),
                  n_matched_sidechain_atoms=len(sc_rows),
                  missing_atoms=sorted(ref_heavy - pred_heavy),
                  extra_atoms=sorted(pred_heavy - ref_heavy),
                  atom_matching='exact atom names; hydrogen/deuterium excluded')

    rotation, shift = np.eye(3), np.zeros(3)
    if align:
        if len(bb_rows) < 3:
            result.update(status='insufficient_backbone_for_alignment',
                          backbone_rmsd=None, sidechain_rmsd=None,
                          all_heavy_rmsd=None)
            return result, None, None
        # fit on the backbone only, then move the whole prediction with it
        rotation, shift = fit_rigid_transform(pred[bb_rows], ref[bb_rows])

    moved = pred @ rotation + shift
    result['backbone_rmsd'] = coordinate_rmsd(ref[bb_rows], moved[bb_rows]) if bb_rows else None
    result['sidechain_rmsd'] = coordinate_rmsd(ref[sc_rows], moved[sc_rows]) if sc_rows else None
    result['all_heavy_rmsd'] = coordinate_rmsd(ref, moved) if len(keys) else None
    if len(keys) == 0:
        result['status'] = 'no_matching_heavy_atoms'
    return result, rotation, shift


# Task: I2
# Author: Yitian Xiao (I2)
def evaluate_candidate_pool(reference_coords, candidate_ids, candidate_coords,
                            selected_id, error_fn, threshold):
    """Score one placed candidate pool against the experimental atoms."""
    reference = np.asarray(reference_coords, dtype=float)
    pool = np.asarray(candidate_coords, dtype=float)
    ids = list(candidate_ids)

    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) == 0:
        raise ValueError('reference atoms must form a nonempty (M, 3) array')
    if len(ids) == 0 or pool.shape != (len(ids), len(reference), 3):
        raise ValueError('candidate coordinates must have shape (K, M, 3)')
    if not np.isfinite(reference).all() or not np.isfinite(pool).all():
        raise ValueError('coordinates must be finite')
    if any(not isinstance(cid, str) or cid == '' for cid in ids):
        raise ValueError('candidate IDs must be nonempty strings')
    if len(set(ids)) != len(ids):
        raise ValueError('candidate IDs must be unique')
    if selected_id not in ids:
        raise ValueError('selected_id is not in the candidate pool')
    if not callable(error_fn):
        raise ValueError('error_fn must be callable')

    limit = np.asarray(threshold, dtype=float)
    if limit.ndim != 0 or not np.isfinite(limit) or limit < 0:
        raise ValueError('threshold must be one finite nonnegative number')

    errors = []
    for xyz in pool:
        value = np.asarray(error_fn(reference, xyz), dtype=float)
        if value.ndim != 0 or not np.isfinite(value) or value < 0:
            raise ValueError('error_fn must return a finite nonnegative number')
        errors.append(float(value))

    oracle_id, oracle_error = choose_oracle(ids, errors)
    selected_error = errors[ids.index(selected_id)]
    label = classify_selection(selected_error, errors, float(limit))
    return dict(selected_id=selected_id,
                selected_error=selected_error,
                oracle_id=oracle_id,
                oracle_error=oracle_error,
                label=label,
                n_candidates=len(ids),
                candidate_errors=errors,
                improvement=selected_error - oracle_error)


# Task: I3
# Author: Yitian Xiao (I3)
def evaluate_stages(reference, python_result, c_result, config):
    """I3 callback: native accuracy and direct Python/C agreement."""
    if c_result.get('status') != 'ok':
        raise ValueError('the C run must succeed before comparing coordinates')

    selected = python_result['selected']
    c_structure = c_result['structure']
    c_map = config.get('c_chain_map')
    py_map = config.get('python_chain_map')

    c_native, _, _ = _compare_structures(reference, c_structure, align=True,
                                        chain_map=c_map)
    py_native, _, _ = _compare_structures(reference, selected, align=True,
                                         chain_map=py_map)

    # rename both sides to the same chains, then compare without fitting anything
    c_keys, c_xyz = _prepare_structure(c_structure, c_map)
    py_keys, py_xyz = _prepare_structure(selected, py_map)
    parity, _, _ = _compare_structures(dict(atom_keys=c_keys, coords=c_xyz),
                                      dict(atom_keys=py_keys, coords=py_xyz),
                                      align=False)

    metrics = dict(
        protein_id=config['protein_id'], chain=config.get('chain'),
        status=c_native['status'],
        c_vs_native_backbone_rmsd_angstrom=c_native['backbone_rmsd'],
        c_vs_native_sidechain_rmsd_angstrom=c_native['sidechain_rmsd'],
        n_matched_backbone_atoms=c_native['n_matched_backbone_atoms'],
        n_matched_sidechain_atoms=c_native['n_matched_sidechain_atoms'],
        python_vs_c_rmsd=parity['all_heavy_rmsd'], python_vs_c=parity,
        c_vs_native=c_native, python_vs_native=py_native,
        exclusions=c_native['missing_atoms'],
        unexpected_atoms=c_native['extra_atoms'])
    if 'corrected' in python_result:
        corrected, _, _ = _compare_structures(reference, python_result['corrected'],
                                             align=True, chain_map=py_map)
        metrics['python_corrected_vs_native'] = corrected
    return metrics


# Task: I3
# Author: Yitian Xiao (I3)
def evaluate_pools(reference, python_result, config):
    """I3 callback: align on the selected backbone once, then call I2 per pool."""
    ref_keys, ref_xyz = _prepare_structure(reference)
    native = dict(zip(ref_keys, ref_xyz))
    selected = python_result['selected']

    _, rotation, shift = _compare_structures(reference, selected, align=True,
                                             chain_map=config.get('python_chain_map'))
    if rotation is None:
        raise ValueError('need matched backbone atoms before scoring candidates')

    threshold = config['threshold']
    scope = config['candidate_scope']
    if scope not in ('nearest_bin', 'retrieved_bins'):
        raise ValueError('candidate_scope must be nearest_bin or retrieved_bins')

    sel_keys, sel_xyz = _prepare_structure(selected, config.get('python_chain_map'))
    placed_selected = dict(zip(sel_keys, sel_xyz))
    chain_map = config.get('python_chain_map') or {}

    rows, seen = [], set()
    for pool in python_result['candidate_pools']:
        residue = tuple(pool['residue_key'])
        if len(residue) != 3:
            raise ValueError('residue_key should have three parts')
        residue = (chain_map.get(residue[0], residue[0]),) + residue[1:]
        if residue in seen:
            raise ValueError('two pools for the same residue')
        seen.add(residue)

        if pool.get('scope') not in ('nearest_bin', 'retrieved_bins'):
            raise ValueError('each pool must declare its scope')
        if scope == 'retrieved_bins' and pool['scope'] != 'retrieved_bins':
            raise ValueError('retrieved_bins scope needs a retrieved-bin pool')
        candidates = pool['candidates']
        if scope == 'nearest_bin' and pool['scope'] == 'retrieved_bins':
            candidates = [cand for cand in candidates if cand['library_order'] == 0]

        row = dict(residue_key=residue, residue_name=pool['residue_name'],
                   scope=scope, selected_id=pool['selected_id'],
                   n_candidates=len(candidates))
        if not candidates:
            row.update(status='excluded', label=None, reason='empty_candidate_pool')
            rows.append(row)
            continue

        names = list(candidates[0]['atom_names'])
        if not names or any(not isinstance(name, str) or name == '' for name in names):
            raise ValueError('atom names must be nonempty strings')
        if len(set(names)) != len(names):
            raise ValueError('atom names must be distinct')
        for name in names:
            if name in ALL_BACKBONE_ATOMS:
                raise ValueError('a pool should only hold side-chain atoms')
            if not _heavy_atom(residue + (name,)):
                raise ValueError('hydrogens are not compared here')

        ids, placed = [], []
        for cand in candidates:
            if list(cand['atom_names']) != names:
                raise ValueError('candidates disagree on atom names or order')
            xyz = np.asarray(cand['global_coords'], dtype=float)
            if xyz.shape != (len(names), 3) or not np.isfinite(xyz).all():
                raise ValueError('bad placed candidate coordinates')
            ids.append(cand['id'])
            placed.append(xyz)

        if len(set(ids)) != len(ids):
            raise ValueError('candidate IDs must be unique')
        if row['selected_id'] not in ids:
            raise ValueError('selected_id is not in the candidate pool')

        keys = [residue + (name,) for name in names]
        missing = [key for key in keys if key not in native]
        if missing:
            row.update(status='excluded', label=None,
                       reason='missing_native_atoms', missing_atoms=missing)
            rows.append(row)
            continue
        if any(key not in placed_selected for key in keys):
            raise ValueError('the selected structure lost a candidate atom')

        saved = np.array([placed_selected[key] for key in keys])
        chosen = placed[ids.index(row['selected_id'])]
        if not np.allclose(saved, chosen, atol=0.0011, rtol=0):
            raise ValueError('selected_id does not match the selected coordinates')

        moved = np.array(placed) @ rotation + shift
        scored = evaluate_candidate_pool([native[key] for key in keys], ids, moved,
                                         row['selected_id'], coordinate_rmsd, threshold)
        row.update(status='valid', **scored)
        rows.append(row)
    return rows
