"""Backbone, side-chain and I1 reconstruction functions."""

from copy import deepcopy
import math
import numpy as np
from numpy.typing import NDArray

from .function import fit_rigid_transform


# Author: Yitian
def _signed_ca_distance(ca1, ca2, ca3, ca4):
    """Distance ca1-ca4, negative when the four CA atoms are left-handed."""
    turn = np.dot(np.cross(ca2 - ca1, ca3 - ca2), ca4 - ca3)
    distance = np.linalg.norm(ca4 - ca1)
    return -distance if turn < 0 else distance


# Author: Yitian
def _extend_ca_trace(ca):
    """Add the two missing CA positions at each end (PULCHRA prepare_rbins).

    The source superimposes the first/second three CA atoms and moves the
    five-atom block, which is a rigid extrapolation of the chain direction.
    """
    ca = np.asarray(ca, dtype=float)
    if len(ca) < 5:
        raise ValueError('PULCHRA needs at least 5 CA atoms per chain')
    extended = np.empty((len(ca) + 4, 3))
    extended[2:-2] = ca
    rotation, shift = fit_rigid_transform(ca[2:5], ca[0:3])
    head = ca[0:5] @ rotation + shift
    extended[0], extended[1] = head[0], head[1]
    rotation, shift = fit_rigid_transform(ca[-5:-2], ca[-3:])
    tail = ca[-5:] @ rotation + shift
    extended[-2], extended[-1] = tail[3], tail[4]
    return extended


# Author: Yitian
def _clamp(value, low, high):
    return max(low, min(high, value))


# Author: Yitian
def _trace_bins(extended):
    """PULCHRA descriptor bins for every four-CA window (n + 1 windows)."""
    bins = []
    for i in range(len(extended) - 3):
        ca1, ca2, ca3, ca4 = extended[i:i + 4]
        r13_1 = np.linalg.norm(ca3 - ca1)
        r13_2 = np.linalg.norm(ca4 - ca2)
        r14 = _signed_ca_distance(ca1, ca2, ca3, ca4)
        bins.append((
            _clamp(int((r13_1 - 4.6) / 0.3), 0, 9),
            _clamp(int((r13_2 - 4.6) / 0.3), 0, 9),
            _clamp(int((r14 + 11.0) / 0.3), 0, 73),
        ))
    return bins


# Author: Yitian
def _lookup_fragment(table, target_bins):
    """Best table fragment; the C loop stops at the sentinel or an exact hit."""
    best_index = None
    best_score = 1000.0
    for index, entry in enumerate(table):
        if entry['bins'][0] < 0:
            break
        score = (abs(entry['bins'][0] - target_bins[0])
                 + abs(entry['bins'][1] - target_bins[1])
                 + 0.2 * abs(entry['bins'][2] - target_bins[2]))
        if score < best_score:
            best_index, best_score = index, score
        if score <= 1e-3:
            break
    if best_index is None:
        raise ValueError('backbone library has no usable fragment')
    return best_index, best_score


# Task: Y7
# Author: Yitian
def rebuild_backbone(ca_record, backbone_library, options):
    """Rebuild N/C/O (and keep CA) from a CA trace using PULCHRA 3.07 tables.

    The window CA[i-2..i+1] selects a fragment that supplies C and O of
    residue i-1 and N of residue i, exactly as rebuild_backbone() in
    pulchra.c. Atoms are never connected across a segment break.
    """
    keys = [tuple(key) for key in ca_record.get('atom_keys', [])]
    names = list(ca_record.get('residue_names', []))
    coords = np.asarray(ca_record.get('coords', np.empty((0, 3))), dtype=float)
    segments = [list(segment) for segment in ca_record.get('segments', [])]
    options = dict(options or {})

    unknown = set(options) - {'include_ca', 'library_version', 'source_commit'}
    if unknown:
        raise ValueError('unsupported options: %s' % sorted(unknown))
    if coords.shape != (len(keys), 3) or len(names) != len(keys):
        raise ValueError('atom_keys, residue_names and coords must agree')

    if not keys:
        return {'atom_keys': [], 'residue_names': [], 'coords': np.empty((0, 3)),
                'segments': [], 'diagnostics': []}
    if not backbone_library:
        raise ValueError('a reviewed backbone library is required')
    if not segments:
        raise ValueError('a nonempty CA record needs explicit segments')
    flat = [row for segment in segments for row in segment]
    if flat != list(range(len(keys))):
        raise ValueError('segments must list the CA rows once, in chain order')

    include_ca = bool(options.get('include_ca', True))
    atom_order = ['N', 'CA', 'C', 'O', 'OXT']
    rebuilt = {}
    diagnostics = []
    out_pieces = []
    out_segments = []

    for segment_number, segment in enumerate(segments):
        if len(segment) < 5:
            for row in segment:
                diagnostics.append({
                    'residue_key': keys[row],
                    'segment': segment_number,
                    'status': 'excluded',
                    'reason': 'PULCHRA needs at least 5 CA atoms per segment',
                })
            continue

        trace = coords[segment]
        extended = _extend_ca_trace(trace)
        bins = _trace_bins(extended)

        for window in range(len(bins)):
            is_proline = window > 0 and names[segment[window - 1]] == 'PRO'
            table_name = 'proline' if is_proline else 'standard'
            table = backbone_library[table_name]
            library_index, score = _lookup_fragment(table, bins[window])
            fragment = table[library_index]['coords']

            rotation, shift = fit_rigid_transform(fragment[:4], extended[window:window + 4])
            placed = fragment @ rotation + shift
            fitted_ca = fragment[:4] @ rotation + shift
            fit_rmsd = float(np.sqrt(np.mean(np.sum(
                (extended[window:window + 4] - fitted_ca) ** 2, axis=1))))

            if window > 0:
                previous = segment[window - 1]
                rebuilt[(previous, 'C')] = placed[4]
                rebuilt[(previous, 'O')] = placed[5]
            if window < len(segment):
                rebuilt[(segment[window], 'N')] = placed[6]
            else:
                rebuilt[(segment[window - 1], 'OXT')] = placed[6]

            diagnostics.append({
                'residue_key': keys[segment[min(window, len(segment) - 1)]],
                'segment': segment_number,
                'window': window,
                'bins': bins[window],
                'library': table_name,
                'library_index': library_index,
                'library_bins': tuple(table[library_index]['bins']),
                'matching_score': score,
                'fit_rmsd': fit_rmsd,
                'status': 'rebuilt',
                'reason': None,
            })

        start = len(out_pieces)
        for row in segment:
            for atom in atom_order:
                if atom == 'CA':
                    if include_ca:
                        out_pieces.append((row, atom))
                elif (row, atom) in rebuilt:
                    out_pieces.append((row, atom))
        out_segments.append(list(range(start, len(out_pieces))))

    out_keys, out_names, out_coords = [], [], []
    for row, atom in out_pieces:
        out_keys.append(keys[row][:3] + (atom,))
        out_names.append(names[row])
        out_coords.append(coords[row] if atom == 'CA' else rebuilt[(row, atom)])

    return {
        'atom_keys': out_keys,
        'residue_names': out_names,
        'coords': np.array(out_coords, dtype=float).reshape(-1, 3),
        'segments': out_segments,
        'diagnostics': diagnostics,
        'library_version': options.get('library_version', backbone_library.get('version')),
        'source_commit': options.get('source_commit'),
    }


# Task: J4
def assemble_candidate_record(residue_key, candidates, selected_id, scope):
    pass


# Task: J5
# Author: Tiffany
def retrieve_rotamers(residue_name: str, descriptor: list[int], library: dict, scope: str) -> list:
    """
    Assumed paramters: 
        - residue_name = residue type (matches 1st index of rot_stat_idx -> error original rot_stat_idx uses ints)
        - descriptor = RBINS (3 floats; from original: bin13_1, bin13_2, bin14; represents 3 C_ALPHA distances)
        - library = {
                rot_stat_idx = list of dict entries in form [residue_name: str, bin1: int, bin2: int, 
                        bin3: int, n_configs: int, pos: int]
                rot_stat_coords = list of list of 3 ints
                heavy_atoms = list of entries in form {residue_name: [atoms]}
                } (don't need to store original nheavy since heavy_atoms already contains info)
        - scope = 'nearest_bin' or 'retrieved_bins"
    Lines covered: 2408 - 2438 (from rebuild_sidechains())
    TODO: 
        - finish error checking (descriptor, library, assumed correct fetching)
        - determine if / understand need to add ['CA'] to atom_names (after writing C5)
        - understand how local_coords is calculated
    """
    VALID_RES_NAMES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", 
                       "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    
    # GLY skipped b/c has no side chain except H; Other residue_names are considered unsupported
    #       rather than parameter errors
    if residue_name == "GLY" or residue_name not in VALID_RES_NAMES:
        return []

    # check that descriptor is correct shape & only contains floats
    if np.array(descriptor, dtype=float).shape != (3,):
        raise ValueError('descriptor must have shape (3,)')

    # check that library contains needed lists, further confirm that lists are in expected format
    expected_keys = ["rot_stat_idx", "rot_stat_coords", "heavy_atoms"]
    for key in expected_keys:
        if key not in library:
            raise ValueError(f"library paramter is missing expected key {key}")

    # set number of rotomers to retrieve, raise error if scope isn't a valid string
    if scope == "nearest_bin":
        num_bins_kept = 1
    elif scope == "retrieved_bins":
        num_bins_kept = 10
    else:
        raise ValueError("scope parameter must be <nearest_bin|retrieved_bins>")

    sorted_rotomers = []

    # get bin ints from descriptor to match original c code
    bin13_1 = descriptor[0]
    bin13_2 = descriptor[1]
    bin14 = descriptor[2]

    # score every entry for residue type, only keep the first 1 or 10 depending on scope
    for i in range(len(library.get("rot_stat_idx"))):
        # from rot_stat_idx.h, rot_stat_idx[][6] = {residue_type -> residue_name, bin indicies [1 - 3], 
        #       stored conformations in bin, where conformations start in rot_stat_coords}
        # note: rot_stat_idx.h ends with entry where [][0] == -1, not needed since can determine end w/ len
        if library.get("rot_stat_idx")[i][0] != residue_name:
            continue

        # calculate hit score (set weights of 1, 1, 0.2 from PULCHRA)
        hit_score = abs(library.get("rot_stat_idx")[i][1] - bin13_1) + abs(library.get("rot_stat_idx")[i][2] - 
                bin13_2) + 0.2 * abs(library.get("rot_stat_idx")[i][3] - bin14)
        
        # if < num_bins_kept rotomers evaled or hit is better than worst hit score rotomer, 
        #       add rotomer to list in correct position
        # ordering only keeps the first rotomer(s) looked at if there's a tie on the boundary of 
        #       keeping a rotomer (in accordance with original code)
        if len(sorted_rotomers) < num_bins_kept or hit_score < sorted_rotomers[-1][0]:
            k = len(sorted_rotomers) - 1
            while(k >= 0 and hit_score < sorted_rotomers[k][0]):
                k -= 1
            k += 1
            sorted_rotomers.insert(k, [hit_score, i, library.get("rot_stat_idx")[i]])

            # remove last rotamer if inserting into full list
            if len(sorted_rotomers) > num_bins_kept:
                del sorted_rotomers[-1]

    candidates = []
    # convert list of rotomers to candidates (id, bin_id, matching_score, atom_names, local_coords, library_order)
    # hit_score, rot_stat_idx index, [residue_name, bin1, bin2, bin3, n_configs, pos]
    for i, hit_score, rot_stat_lib_idx, rot_stat_idx in enumerate(sorted_rotomers):
        n_configs, coords_start_idx = rot_stat_idx[4], rot_stat_idx[5]

        for config_num in range(n_configs):
            # added 'CA' to beginning of atom_names by AI suggestion after analyzing original 
            #       rot_stat_coords, to update after writing C5
            atom_names = ['CA'] + library.get("heavy_atoms").get(residue_name)

            # AI used to figure out how to calculate local_coords
            #   local_coords = len(atom_names) x 3 matrix
            start = int(coords_start_idx) + len(atom_names) * config_num
            
            candidates.append({
                'id': residue_name + ":" + rot_stat_lib_idx + ":" + config_num, 
                'bin_id': list(rot_stat_idx[1], rot_stat_idx[2], rot_stat_idx[3]), 
                'matching_score': hit_score, 
                'atom_names': atom_names, 
                'local_coords': library.get("rot_stat_coords")[start:start + len(atom_names)].copy(), 
                'library_order': i
            })

    return candidates

# Task: J6
# Author: Tiffany
def place_candidate(local_coords: list[list[int]], origin: list[int], axes: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Assumed paramters: 
        - local_coords = list of side chain atom coords for a single candidate from J5
        - origin = C_ALPHA 3D position (list length 3)
        - axes = frame vectors (3x3 matrix, created from lines 2370 - 2404)
    Lines covered (guessed): 2501 - 2525 (from rebuild_sidechains())
    TODO: 
        - type error checking (checking that elements of lists are floats)
        - consider parallelizing shifting coords (unsure since max atoms placed = 10)
    """
    ACCEPTED_FLOATING_ERROR = 1e-8
    # type error checking

    # paramter shape error checking using np.array (written using AI)
    if np.array(local_coords, dtype=float).ndim != 2 or np.array(local_coords, dtype=float).shape[1] != 3:
        raise ValueError('local_coords must have shape (M, 3)')
    if np.array(origin, dtype=float).shape != (3,):
        raise ValueError('origin must have shape (3,)')
    if np.array(axes, dtype=float).shape != (3, 3):
        raise ValueError('axes must have shape (3, 3)')

    # return early if local_coords is empty assume side-chain has no atoms -> return array sized (0, 3)
    if np.array(local_coords, dtype=float).shape[0] == 0:
        return np.array([]).reshape(0, 3)

    # check that axes are orthonormal (suggested by AI, but implemented without help)
    # check that each axes is unit length (magnitude = 1)
    for i in range(3):
        magnitude = math.sqrt(axes[i][0]**2 + axes[i][1]**2 + axes[i][2]**2)
        # suggested by AI to check using this method to avoid floating point arithmetic errors
        if abs(magnitude - 1.0) > ACCEPTED_FLOATING_ERROR:
            raise ValueError(f"axes[{i}] does not have a magnitude of 1")

    # check that each axes is orthogonal to all other axes (dot product = 0)
    dot0_1 = axes[0][0] * axes[1][0] + axes[0][1] * axes[1][1] + axes[0][2] * axes[1][2]
    dot1_2 = axes[1][0] * axes[2][0] + axes[1][1] * axes[2][1] + axes[1][2] * axes[2][2]
    dot0_2 = axes[0][0] * axes[2][0] + axes[0][1] * axes[2][1] + axes[0][2] * axes[2][2]
    if abs(dot0_1 - 0.0) > ACCEPTED_FLOATING_ERROR or abs(dot1_2 - 0.0) > ACCEPTED_FLOATING_ERROR or (
            abs(dot0_2 - 0.0) > ACCEPTED_FLOATING_ERROR):
        raise ValueError("axes are not orthogonal")

    # check that axes only rotate -> check that each det(axes) = 1 using cofactor expansion
    # (suggested by AI, but implemented without help)
    det = (
        axes[0][0] * (axes[1][1] * axes[2][2] - axes[1][2] * axes[2][1]) - 
        axes[0][1] * (axes[1][0] * axes[2][2] - axes[1][2] * axes[2][0]) +
        axes[0][2] * (axes[1][0] * axes[2][1] - axes[1][1] * axes[2][0])
           )
    if abs(det - 1.0) > ACCEPTED_FLOATING_ERROR:
        raise ValueError('Given axes do not conserve magnitude of multiplied vectors (determinant != 1)')

    # need to reimplement since "local_coords @ axes.T + origin" relies on underlying C code to optimize
    # return local_coords @ axes.T + origin
    placed_coords = []
    # for each atom in the side-chain: rotate relative coords from local_coords using axes and 
    #   shift to C_ALPHA location using origin
    for x, y, z in local_coords:
        new_x = x * axes[0][0] + y * axes[0][1] + z * axes[0][2] + origin[0]
        new_y = x * axes[1][0] + y * axes[1][1] + z * axes[1][2] + origin[1]
        new_z = x * axes[2][0] + y * axes[2][1] + z * axes[2][2] + origin[2]
        placed_coords.append([new_x, new_y, new_z])

    # convert to array according to expected output
    return np.array(placed_coords)


# Task: J7
def capture_candidate_trace(engine, ca_record, options, scope):
    pass


# Task: C5
def select_original_candidate(candidates, context, options):
    pass


# Task: C7
def correct_structure_stage(structure: dict, candidates: list, options: dict[str,bool]) -> dict:
    pass


# Task: I1
# Author: Jimmy
def run_reconstruction(ca_record, options, stages):
    for name in ('optimize_ca', 'backbone', 'candidates', 'select', 'correct'):
        if not callable(stages.get(name)):
            raise ValueError(f'Missing reconstruction stage: {name}')

    # Each callback gets a working copy so later stages cannot overwrite earlier results.
    original = deepcopy(ca_record)
    optimized = stages['optimize_ca'](deepcopy(original), deepcopy(options))
    backbone = stages['backbone'](deepcopy(optimized), deepcopy(options))
    candidates = stages['candidates'](deepcopy(optimized), deepcopy(backbone), deepcopy(options))
    if any('selected_id' in pool for pool in candidates):
        raise ValueError('Candidate generation must leave the selection open')

    selected = stages['select'](deepcopy(backbone), deepcopy(candidates), deepcopy(options))
    pools = deepcopy(candidates)
    for pool, chosen in zip(pools, selected['selected_candidate_ids'], strict=True):
        ids = pool.get('candidate_ids')
        if ids is None:
            ids = [candidate['id'] for candidate in pool['candidates']]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate candidate ID')
        if chosen not in (ids or [None]):
            raise ValueError('Selected candidate is outside the pool')
        pool['selected_id'] = chosen

    corrected = stages['correct'](deepcopy(selected), deepcopy(pools), deepcopy(options))

    return {
        'ca_input': original,
        'ca_optimized': optimized,
        'backbone': backbone,
        'candidate_pools_before_selection': candidates,
        'candidate_pools': pools,
        'selected': selected,
        'corrected': corrected,
        'options': deepcopy(options),
        'status': 'ok',
    }
