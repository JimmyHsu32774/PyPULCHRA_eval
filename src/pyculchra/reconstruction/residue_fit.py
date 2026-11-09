<<<<<<< HEAD
import copy
from itertools import combinations
=======
>>>>>>> 108116a5ac53390783f474a3239fb0d1251b97de
import math
import numpy as np
from scipy.spatial import KDTree

<<<<<<< HEAD
# Author: Caroline Rhoads (C1)
=======
#Author: Caroline Rhoads (C1)
>>>>>>> 108116a5ac53390783f474a3239fb0d1251b97de
def find_close_pairs(coords: list[list[float]], cutoff: float, excluded_pairs: set[tuple[int, int]]) -> np.ndarray:
    """C1: Implement the contract above; document your reasoning.
    Implemented by: Caroline Rhoads
    
    Inputs:
        coords: list[list[float]] = a list of 3D coordinates for atoms, where each inner list contains three floats representing the x, y, and z coordinates of an atom.
        cutoff: float = the distance threshold below which atoms are considered to be in close proximity.
        excluded_pairs: set[tuple[int, int]] = a set of atom index pairs that should be excluded from consideration.
    Outputs:
        clash_arr: ndarray = an array of shape (m, 2) containing the indices of two atoms that are closer than the cutoff distance and are not in the excluded_pairs set.
    Reasoning:
        1. Input validation: The function first checks if the cutoff is a finite positive number and if the coords input is a 2D array with shape (n, 3) and contains finite values. If any of these checks fail, a ValueError is raised.
        2. Distance calculation: The function iterates through all unique pairs of atoms (i, j) where i < j. For each pair, it checks if the pair is in the excluded_pairs set.
        3. If the pair is not excluded, it calculates the squared distance between the two atoms using the Euclidean distance formula. If the squared distance is less than the square of the cutoff, the pair is added to the clash_list.
        4. Finally, the function returns the clash_list containing all pairs of atoms that are closer than the cutoff distance and not excluded.
    """
    
    # Input validation:
       
    try:
        cutoff = float(cutoff)
    except (ValueError, TypeError) as error:
        raise ValueError("cutoff passed into find_close_pairs couldn't be converted into float type") from error
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff passed into find_close_pairs is not a finite positive number")
    
    arr = np.asarray(coords, dtype = float)    #np.asarray() doesn't copy the data over if it's already in an array which we won't need
    
    if arr.ndim != 2:
        raise ValueError("the dimension of the atom list passed into find_close_pairs is not 2")
    if arr.shape[1] != 3:
        raise ValueError("the shape of the atom list passed into find_close_pairs is not n x 3")
    if not np.isfinite(arr).all():
        raise ValueError("the entries in the atom list passed into find_close_pairs must be finite")

    # build sciPy KDTree for efficient search using query_pairs()
    tree = KDTree(arr)
    clash_arr = tree.query_pairs(r = cutoff, output_type='ndarray') # all other parameters are default, this stores the output as an ndarray of shape (m, 2) where m is the number of pairs found
    if len(clash_arr) == 0:
        return np.empty((0, 2), dtype=int) # if no pairs are found, return an empty array of shape (0, 2)

    # query_pairs() uses <= cutoff, but we want < cutoff, so we filter out the pairs that are exactly at the cutoff distance:
    angstroms = np.linalg.norm(arr[clash_arr[:, 0]] - arr[clash_arr[:, 1]], axis=1) # finds euclidean distance between each pair of atoms in clash_arr
    clash_arr = clash_arr[angstroms < cutoff]   # returns a list of bools aligned with clash_arr.
    
    # Remove excluded pairs
    clash_arr = np.array([
        pair for pair in clash_arr
        if (pair[0], pair[1]) not in excluded_pairs
        and (pair[1], pair[0]) not in excluded_pairs
    ]).reshape(-1, 2)

<<<<<<< HEAD
    return clash_arr


# Implementation author: Caroline Rhoads
def build_excluded_pairs(structure: dict) -> set:
    """Build the set of atom pairs to exclude from clash detection."""

    excluded_pairs = set()

    # group the atoms by residue and neighboring residue:
    # this creates a dictionary of residues and their corresponding atom index list
    residue_ind = {}
    for atom_id, (chain, resseq, icode, name) in enumerate(structure['atom_keys']):
        residue_ind.setdefault((chain, resseq, icode), []).append((atom_id, name))

    # add combinations of the atom pairs in each residue to the excluded list:
    for atoms in residue_ind.values():
        atom_ids = [atom_id for atom_id, _ in atoms]
        excluded_pairs.update(combinations(atom_ids, 2))

    # add combinations of the backbone atoms in neighboring residues to the excluded list:
    for res1, res2 in zip(list(residue_ind), list(residue_ind)[1:]):
        if res1[0] != res2[0]:  #checks chain number
            continue
        if res2[1] < res1[1]:
            raise ValueError("residues are not in increasing order within a chain")
        backbone1 = {name: atom_id for atom_id, name in residue_ind[res1] if name in ('N', 'CA', 'C', 'O')}
        backbone2 = {name: atom_id for atom_id, name in residue_ind[res2] if name in ('N', 'CA', 'C', 'O')}
        if 'C' in backbone1 and 'N' in backbone2:
            excluded_pairs.add((backbone1['C'], backbone2['N']))
        if 'CA' in backbone1 and 'CA' in backbone2:
            excluded_pairs.add((backbone1['CA'], backbone2['CA']))

    return excluded_pairs


# Implementation author: Caroline Rhoads
def correct_structure_stage(structure: dict, candidates: list, options: dict[str,bool]) -> dict:
#def correct_structure_stage({'atom_keys': [('A',1,'','CA')], 'residue_names': ['ALA'], 'coords': np.array([[0.,0.,0.]])}, [], {'correct_sidechains': False})) -> dict:

    """C7: Implement the contract above; no reference solution is supplied.
    
    What this does:
        if options['correct_sidechains'] is True, then for each candidate in candidates, the function will correct the sidechain conformations of the structure based on the candidate's information. The corrected structure will be returned as a new dictionary.
            else, if options['correct_sidechains'] is False, the function will return the original structure without any modifications.
        stage_status is a reciept of the action taken, this is set to 'disabled' if the user is skipping this step and a copy of the structure is returned, this is set to 'corrected' if the sidechains were corrected..

    """
    CLASH_CUTOFF = 2.0  # example cutoff distance for clashes
    MAX_ITERATIONS = 100  # example maximum number of iterations for correction

    # Check inputs are good to go:
    if not isinstance(options, dict):
        raise ValueError("correct_structure_stage() input 'options' must be a dictionary")
    if 'correct_sidechains' not in options:
        raise ValueError("correct_structure_stage() input 'options' must include the key 'correct_sidechains'")
    if not isinstance(options['correct_sidechains'], bool):
        raise ValueError("correct_structure_stage() input 'options' must have 'correct_sidechains' set to a boolean value")

    # first branch, do nothing and return a copy of the structure with stage_status set to 'disabled':
    if options['correct_sidechains'] is False:  
        # Create a copy of the structure to avoid modifying the original
        return_structure = copy.deepcopy(structure)
        return_structure['stage_status'] = 'disabled'
        return return_structure

    # Implementation:

    if options['correct_sidechains'] is True:
        # Create a copy of the structure to avoid modifying the original
        return_structure = copy.deepcopy(structure)

        # build exclusion set of pairs to exclude these clashes
        excluded_pairs = build_excluded_pairs(return_structure)
        
        # get the list of pairs that is clashing:
        coords = return_structure['coords']

        clashes = find_close_pairs(coords, CLASH_CUTOFF, excluded_pairs)

        # MAIN LOOP BEGINS TO CHECK AND ROTATE RESIDUES:
        i = 0
        while len(clashes) > 0 and i < MAX_ITERATIONS:
            i += 1

            # organize clashes by residue... 
            clashes_by_residue = {}
            atom_keys = return_structure['atom_keys']

            for a, b in clashes.tolist():
                for atom in (a, b):
                    residue_key = atom_keys[atom][:3]     # (chain, resseq, icode)
                    clashes_by_residue.setdefault(residue_key, set()).add(atom)

            # Rotate residues with the most clashes to next best conformation from library:
            for residue in clashes_by_residue:
                #TODO
                # Implement the logic to rotate the sidechain of the residue to the next best conformation from the library
                # This is a placeholder for the actual implementation
                pass


            # recheck for clashes after updating the structure
            new_coords = return_structure['coords']
            clashes = find_close_pairs(new_coords, CLASH_CUTOFF, excluded_pairs)
        
        return_structure['stage_status'] = 'corrected'
        return return_structure

    raise NotImplementedError("enabled branch is BLOCKED: needs candidate format, library data, C reference trace")
=======
    return clash_arr
>>>>>>> 108116a5ac53390783f474a3239fb0d1251b97de
