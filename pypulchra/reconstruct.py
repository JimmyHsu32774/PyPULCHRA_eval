"""Backbone, side-chain and I1 reconstruction functions."""

from copy import deepcopy
import math
import numpy as np
from numpy.typing import NDArray

from .clashes import build_excluded_pairs, find_close_pairs

from .function import fit_rigid_transform, make_local_frame

BACKBONE_NAMES = {'N', 'CA', 'C', 'O', 'OXT'}


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
# Author: Tiffany; tiny edit by Caroline
def assemble_candidate_record(residue_key, candidates, selected_id, scope):
    """
    Assemble one JSON ready candidate record containing residue key, scope, list of all ids, coords,
    and selected ID if applicable. 
    """
    # check invalid scope
    if scope != "nearest_bin" and scope != "retrieved_bins":
        raise ValueError("scope parameter must be <nearest_bin|retrieved_bins>")

    candidate_record = {
        "residue_key": list(residue_key),
        "scope": scope
    }

    # check that selected_id isn't None, still valid since GLY and unsupported residues don't have candidates
    if selected_id == None:
        candidate_record["selected_index"] = None   

    # get list of ids and coords, check for NaN or inconsistant atom count, check for duplicate ids
    ids = []
    all_coords = []
    for candidate in candidates:
        # check that candidate dict has expected keys
        if "id" not in candidate or "local_coords" not in candidate:
            raise ValueError("Candidate entry doesn't contain expected keys")

        id = candidate.get("id")
        coords = candidate.get("local_coords")

        if id in ids:
            raise ValueError(f"Duplicate {id} found in candidate ids")

        # AI-assisted
        try:
            arr = np.array(coords, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"local_coords for {id} must be a numeric (M, 3) array")
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            raise ValueError(f"local_coords for {id} must have shape (M, 3) with M >= 1")
        
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"local_coords for {id} contain NaN or infinity")

        ids.append(id)
        all_coords.append(arr)

    candidate_record["candidate_ids"] = ids

    # only determine if not already set to None, check if missing selected ID
    if "selected_index" not in candidate_record:
        try:
            candidate_record["selected_index"] = candidate_record.get("candidate_ids").index(selected_id)
        except ValueError:
            raise ValueError("List of candidate ids is missing selected ID")
        
    candidate_record["coordinates"] = [arr.tolist() for arr in all_coords]

    # Added by Caroline to assist with correct_structure_stage... (atom_names needs to be carried over to match each coordinate row to an atom)
    if len(candidates) > 0 and "atom_names" in candidates[0]:
        candidate_record["atom_names"] = list(candidates[0]["atom_names"])

    return candidate_record


# Task: J5
# Author: Tiffany
def retrieve_rotamers(residue_name: str, descriptor: list[int], library: dict, scope: str) -> list:
    """
    Retrive the list of rotomer candidates for a residue based on bin distances, number of candidates
    retrived depends on scope

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
    """
    VALID_RES_NAMES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", 
                       "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    
    # GLY skipped b/c has no side chain except H; Other residue_names are considered unsupported
    #       rather than parameter errors
    if residue_name == "GLY" or residue_name not in VALID_RES_NAMES:
        return []

    # check that descriptor is correct shape & only contains floats
    try:
        descriptor_array = np.asarray(descriptor, dtype=float)
    except (TypeError, ValueError):
        raise ValueError('descriptor must have shape (3,)')
    if descriptor_array.shape != (3,) or not np.all(np.isfinite(descriptor_array)):
        raise ValueError('descriptor must have shape (3,)')
    descriptor = descriptor_array

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
    for library_order, (hit_score, rot_stat_lib_idx, rot_stat_idx) in enumerate(sorted_rotomers):
        n_configs, coords_start_idx = rot_stat_idx[4], rot_stat_idx[5]

        for config_num in range(n_configs):
            atom_names = library["heavy_atoms"].get(residue_name)
            if atom_names is None:
                raise ValueError(f"library is missing heavy atoms for {residue_name}")
            atom_names = list(atom_names)

            # AI-assisted
            has_ca_anchor = library.get("includes_ca_anchor", False)
            conformer_stride = len(atom_names) + (1 if has_ca_anchor else 0)
            start = int(coords_start_idx) + conformer_stride * config_num
            if has_ca_anchor:
                start += 1
            
            candidates.append({
                'id': f"{residue_name}:{rot_stat_lib_idx}:{config_num}",
                'bin_id': [rot_stat_idx[1], rot_stat_idx[2], rot_stat_idx[3]],
                'matching_score': hit_score, 
                'atom_names': atom_names, 
                'local_coords': np.asarray(library["rot_stat_coords"][start:start + len(atom_names)]).copy(),
                'library_order': library_order
            })

    return candidates

# Task: J6
# Author: Tiffany
def place_candidate(local_coords: list[list[int]], origin: list[int], axes: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Converts from local coords to global coords based on given C-alpha origin and axes

    Assumed paramters: 
        - local_coords = list of side chain atom coords for a single candidate from J5
        - origin = C_ALPHA 3D position (list length 3)
        - axes = frame vectors (3x3 matrix, created from lines 2370 - 2404)
    Lines covered (guessed): 2501 - 2525 (from rebuild_sidechains())
    TODO: 
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
# Author: Tiffany
def select_original_candidate(candidates, context, options):
    """
    Returns PULCHRA selection for candidate from candidate pool based on options

    Assumed paramters: 
        - candidates = {
                id = residue_name + ":" + rot_stat_lib_idx + ":" + config_num
                bin_id = list(rot_stat_idx[1], rot_stat_idx[2], rot_stat_idx[3])
                matching_score = hit_score
                atom_names = ['CA'] + library.get("heavy_atoms").get(residue_name)
                local_coords = start = the local coords of heavy atoms relative to the origin
                library_order = rotomer # from sorted rotomer list
                global_coords = the global x, y, z coords of heavy atoms from local_coords (from J6)
            }
        - context = {
                origin = global x, y, z coords of C-alpha backbone position
            }
        - options = {
                pdb_sg = look at every member of the 2 best-ranked bins (l < 2); for each, take the
                    centroid of the CA anchor point plus its side-chain atoms (nsc points) and its
                    SQUARED distance to the residue's reference centroid; keeps candidate w/
                    smallest dist. In case of ties, keeps earlier candidate. Has a starting
                    starting best dist of 100.0, but if nothing has dist < 100.0, keeps "library 
                    position 0"
            }
    TODO: 
        - error checking
    """
    if candidates is None or len(candidates) == 0:
        return {'selected_id': None, 'decision_trace': {'status': 'no_candidates'}}

    # error check for duplicates

    if not options['pdb_sg']:
        first_candidate = candidates[0]
        return {
            'selected_id': first_candidate['id'],
            'decision_trace': {
                'status': 'selected',
                'rule': 'first_member_of_best_ranked_bin',
                'pdb_sg': False,
                'bin_id': first_candidate['bin_id'],
                'matching_score': first_candidate['matching_score'],
                'n_candidates_in_pool': len(candidates)
            }
        }

    # following code is AI-assisted
    # convert to array to make later computations easier
    ca_position = np.asarray(context['origin'], dtype=float)
 
    best_candidate = None
    INITIAL_BEST_SQ_DIST = 100.0
    best_sq_dist = INITIAL_BEST_SQ_DIST
    tied_candidate_ids = []
    scanned_candidates = []

    # only compare first 2 candidates in accordance w/ original code, may modify later using options
    for candidate in candidates[:2]: 
        side_chain_coords = np.asarray(candidate['global_coords'], dtype=float)
        points_in_centroid = len(side_chain_coords) + 1   # side-chain atoms + CA anchor point
        # find center by avg coords by num of side chain atoms
        candidate_centroid = (ca_position + side_chain_coords.sum(axis=0)) / points_in_centroid
        squared_distance = float(np.sum((ca_position - candidate_centroid) ** 2))
        scanned_candidates.append({'id': candidate['id'], 'bin_id': candidate['bin_id'],
                                   'squared_distance': squared_distance})
 
        if squared_distance < best_sq_dist:
            best_sq_dist = squared_distance
            best_candidate = candidate
            tied_candidate_ids = [candidate['id']]
        elif squared_distance == best_sq_dist and best_candidate is not None:
            tied_candidate_ids.append(candidate['id'])
 
    decision_trace = {
        'rule': 'closest_centroid_in_two_best_ranked_bins',
        'pdb_sg': True,
        'scanned_candidates': scanned_candidates,
        'initial_best_squared_distance': INITIAL_BEST_SQ_DIST,
        'n_candidates_in_pool': len(candidates),
    }

    # highly unlikely to happen, but include if it does happen
    if best_candidate is None:
        decision_trace['status'] = 'fallback_no_candidate_below_initial_distance'
        decision_trace['note'] = 'C keeps library position 0 here; it may not be in this pool'
        return {'selected_id': None, 'decision_trace': decision_trace}
 
    decision_trace['status'] = 'selected'
    decision_trace['best_squared_distance'] = best_sq_dist
    decision_trace['tied_candidate_ids'] = tied_candidate_ids   # first one wins (strict '<')
    return {'selected_id': best_candidate['id'], 'decision_trace': decision_trace}


# Task: Helper function for C7:
# Author: Caroline
def _find_clashes(coords, atom_keys, excluded_pairs, clash_cutoff) -> list[tuple[int, int]]:
    """
    Almost the same as find_close_pairs() (in the file clashes.py) but it filters out backbone-backbone clashes."""

    clashes = find_close_pairs(coords, clash_cutoff, excluded_pairs)

    clashes_filtered = []
    for atom_1, atom_2 in clashes:
        # Exclude backbone-backbone clashes
        if atom_keys[atom_1][3] in BACKBONE_NAMES and atom_keys[atom_2][3] in BACKBONE_NAMES:
            continue
        clashes_filtered.append((atom_1, atom_2))

    return clashes_filtered


# Task: Helper function for C7:
# Author: Caroline
def _count_clashes(residue_key, atom_keys, clash_list) -> int:
    """
    Count the number of clashes that involve atoms from a specific residue.
    """
    count = 0
    # if either atom in a clash pair belongs to this residue, += 1
    for atom_1, atom_2 in clash_list:
        if tuple(atom_keys[atom_1][:3]) == residue_key:
            count += 1
        if tuple(atom_keys[atom_2][:3]) == residue_key:
            count += 1
    return count


# Task: C7
# Author: Caroline
def correct_structure_stage(structure: dict, candidates: list, options: dict[str,bool]) -> dict:
    #def correct_structure_stage({'atom_keys': [('A',1,'','CA')], 'residue_names': ['ALA'], 'coords': np.array([[0.,0.,0.]])}, [], {'correct_sidechains': False})) -> dict:

    """C7: Implement the contract above; no reference solution is supplied.
    
    What this does:
        if options['correct_sidechains'] is True, then for each candidate in candidates, the function will correct the sidechain conformations of the structure based on the candidate's information. The corrected structure will be returned as a new dictionary.
            else, if options['correct_sidechains'] is False, the function will return the original structure without any modifications.
        stage_status is a reciept of the action taken, this is set to 'disabled' if the user is skipping this step and a copy of the structure is returned, this is set to 'corrected' if the sidechains were corrected..

    """
    CLASH_CUTOFF = 2.0  # example cutoff distance for clashes, C code used 1.6
    MAX_ITERATIONS = 3  # number of iterations for sidechain correction from C code, = 100 is for backbone not sidechain checks

    # Check inputs are good to go:
    if not isinstance(options, dict):
        raise ValueError("correct_structure_stage() input 'options' must be a dictionary")
    if 'correct_sidechains' not in options:
        raise ValueError("correct_structure_stage() input 'options' must include the key 'correct_sidechains'")
    if not isinstance(options['correct_sidechains'], bool):
        raise ValueError("correct_structure_stage() input 'options' must have 'correct_sidechains' set to a boolean value")

    # Create a copy of the structure to avoid modifying the original
    return_structure = deepcopy(structure)
    
    # first branch, do nothing and return a copy of the structure with stage_status set to 'disabled':
    if options['correct_sidechains'] is False:  
        return_structure['stage_status'] = 'disabled'
        return return_structure

    # Implementation:
    if options['correct_sidechains'] is True:

        atom_keys = return_structure['atom_keys']   # (chain, resseq, icode, atom_name)
        residue_names = return_structure['residue_names']   # the 3 letter code for each residue
        coords = np.array(return_structure['coords'], dtype=float)   # copies the coordinates of the whole structure

        # If the structure is empty:
        if len(atom_keys) == 0:
            return_structure['stage_status'] = 'corrected'
            return_structure['diagnostics'] = []
            return_structure['remaining_clashes'] = 0
            return return_structure

        # Set up data structures to rotate residues:
        # Group atoms by residue:
        residue_atoms = {}  # build a dictionary that lists the atom IDs for each residue
        ca_id = {}  # build a dictionary that lists the CA atom ID for each residue for easy lookup later
        chain_ca = {}  # build a dictionary that lists the CA atom IDs for each chain for easy lookup later
        for atom_id, key in enumerate(atom_keys):
            residue_key = tuple(key[:3])  # (chain, resseq, icode)
            residue_atoms.setdefault(residue_key, []).append(atom_id)   # group atom IDs by residue
            if key[3] == 'CA':
                ca_id[residue_key] = atom_id    # don't need setdefault, there should only be one CA atom per residue
                chain_ca.setdefault(key[0], []).append(residue_key)   # group CA atom IDs by chain

        # Build the frame's dictionary (frame is the coordinate system of the ca for each residue's sidechain placement):
        frames = {}
        for _, residue_list in chain_ca.items():
            if len(residue_list) < 5:
                continue  # skip chains with less than 5 residues, _extend_ca_trace() needs 5 CA atoms

            # Get a list of ca coordinates from ca_id list
            ca_coords = []
            for residue_key in residue_list:
                ca_coords.append(coords[ca_id[residue_key]])
            ca_coords = np.array(ca_coords)

            # add two extra CA positions at each end of the chain:
            extended_ca_coords = _extend_ca_trace(ca_coords)

            for i, residue_key in enumerate(residue_list):
                # extended[i + 3] corresponds to the next CA's residue
                # extended[i + 2] corresponds to the current CA's residue
                # extended[i + 1] corresponds to the previous CA's residue
                origin, axes = make_local_frame(extended_ca_coords[i + 1], extended_ca_coords[i + 3], extended_ca_coords[i + 2]) # call is previous, next, current
                frames[residue_key] = np.column_stack((axes[:, 0], axes[:, 2], -axes[:, 1]))


        # build the candidate pool for each residue:
        candidate_pools = {}
        for pool in candidates:
            candidate_pools[tuple(pool['residue_key'])] = pool

        # Clashes check:
        excluded_pairs = build_excluded_pairs(return_structure) # build the set of bonded pairs and sidechain pairs to exclude from clashes
        clashes = _find_clashes(coords, atom_keys, excluded_pairs, CLASH_CUTOFF)    # returns list of tuple clashes excluding backbone-backbone clashes

        # start of log of changed residues:
        rotated_residues = []


        # MAIN LOOP BEGINS TO CHECK SIDE-CHAIN CLASHES AND GREEDILY ROTATES RESIDUES:
        i = 0
        while len(clashes) > 0 and i < MAX_ITERATIONS:
            i += 1

            # Clashing residues are rotated in chain order:
            for residue_key, atom_list in residue_atoms.items():
                # glycine has no sidechain, so skip it:
                if residue_names[atom_list[0]] == 'GLY':
                    continue
                # skip if no candidate or no frame:
                if residue_key not in candidate_pools or residue_key not in frames:
                    continue

                # NOTE: You have to do this inside the loop not before because the clashes of other residues may change while rotating this residue...
                count_before_rotation = _count_clashes(residue_key, atom_keys, clashes)
                if count_before_rotation == 0:
                    continue

                candidate_pool = candidate_pools[residue_key]
                # sidechain_atoms = [atom_keys[atom_id][3] for atom_id in atom_list if atom_keys[atom_id][3] not in BACKBONE_NAMES]  
                sidechain_atoms = {}    # get sidechain atom names
                for atom_id in atom_list:
                    name = atom_keys[atom_id][3]
                    if name not in BACKBONE_NAMES:
                        sidechain_atoms[name] = atom_id

                # need the atom name for each row of a candidates coords
                names = candidate_pool.get('atom_names') or list(sidechain_atoms)  # use candidate atom names if available, otherwise use the sidechain atom names from the structure
                if sorted(names) != sorted(sidechain_atoms):
                    raise ValueError(f"Candidate atom names {names} do not match sidechain atom names {sidechain_atoms} for residue {residue_key}")

                # NOW WE TRY CANDIDATES, KEEPING THE ONE THAT REDUCES CLASHES THE MOST:
                best_count = None
                best_id = None
                best_coords = None

                for candidate_id, local_coords in zip(candidate_pool['candidate_ids'], candidate_pool['coordinates']):
                    if len(names) != len(local_coords):
                        raise ValueError(f"Candidate {candidate_id} has {len(local_coords)} coordinates but expected {len(names)} based on atom names for residue {residue_key}")

                    # get the global coords:
                    placement = place_candidate(local_coords, coords[ca_id[residue_key]], frames[residue_key])

                    # trial placement:
                    for name, position in zip(names, placement):
                        coords[sidechain_atoms[name]] = position  # update the coords for this sidechain atom
                    trial_clashes = _find_clashes(coords, atom_keys, excluded_pairs, CLASH_CUTOFF)
                    trial_count = _count_clashes(residue_key, atom_keys, trial_clashes)

                    # place the candidate if its the best so far, strict '<' means first candidate wins ties
                    if best_count is None or trial_count < best_count:
                        best_count = trial_count
                        best_id = candidate_id
                        best_coords = placement

                    if trial_count == 0:
                        break  # no need to try more candidates if we found one with no clashes

                # empty candidate pool:
                if best_count is None:
                    continue  # no candidates to try, skip to next residue

                # make sure the best candidate is written back into the structure's coords:
                for name, position in zip(names, best_coords):
                    coords[sidechain_atoms[name]] = position  # update the coords for this sidechain atom

                # update the log:
                rotated_residues.append({
                    'residue_key': residue_key,
                    'pass': i,
                    'selected_candidate_id': best_id,
                    'clash_count_before': count_before_rotation,
                    'clash_count_after': best_count
                })

                # update the clash list:
                clashes = _find_clashes(coords, atom_keys, excluded_pairs, CLASH_CUTOFF)

        # finalize the return structure:
        return_structure['coords'] = coords
        return_structure['stage_status'] = 'corrected'
        return_structure['diagnostics'] = rotated_residues
        return_structure['remaining_clashes'] = len(clashes)
        return return_structure


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
