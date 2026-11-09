from copy import deepcopy
import math
import numpy as np


# Author: Jimmy
def get_ca_energy_and_direction(xyz, original_xyz, cispro, max_shift=3.0, *, calc_gradient=True):
    n = len(xyz)
    direction = np.zeros((n, 3)) if calc_gradient else None
    energy = 0.0

    for i in range(n):
        diff_orig = xyz[i] - original_xyz[i]
        dist_orig = np.linalg.norm(diff_orig)
        if dist_orig > max_shift:
            energy += 0.01 * (dist_orig ** 2)
            if calc_gradient:
                direction[i] -= 0.02 * diff_orig

        if i > 0:
            diff = xyz[i] - xyz[i - 1]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                dist = 1e-12

            target_len = 2.9 if cispro[i] else 3.8
            err = target_len - dist
            energy += 10.0 * (err ** 2)
            if calc_gradient:
                force = 20.0 * err * (diff / dist)
                direction[i] += force
                direction[i - 1] -= force

        for j in range(i - 2):
            diff = xyz[i] - xyz[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                dist = 1e-12

            if dist < 3.5:
                energy += 10.0 * (dist ** 2)
                if calc_gradient:
                    repulsion = -80.0 * (dist - 3.5) * (diff / dist)
                    direction[i] += repulsion
                    direction[j] -= repulsion

        if 0 < i < n - 1:
            vec1 = xyz[i - 1] - xyz[i]
            vec2 = xyz[i + 1] - xyz[i]
            len1 = np.linalg.norm(vec1)
            len2 = np.linalg.norm(vec2)

            cos_val = np.dot(vec1, vec2) / (len1 * len2)
            cos_val = max(-1.0, min(1.0, cos_val))
            angle = math.acos(cos_val)

            min_angle = math.radians(80)
            max_angle = math.radians(150)
            target_angle = max(min_angle, min(max_angle, angle))

            angle_err = angle - target_angle
            energy += 20.0 * (angle_err ** 2)
            if not calc_gradient:
                continue

            sin_val = math.sqrt(max(0.0, 1.0 - cos_val ** 2))
            if sin_val < 1e-12:
                sin_val = 1e-12

            scale = 40.0 * angle_err / sin_val
            grad_a = scale * (vec2 / (len1 * len2) - cos_val * vec1 / (len1 ** 2))
            grad_b = scale * (vec1 / (len1 * len2) - cos_val * vec2 / (len2 ** 2))

            direction[i - 1] += grad_a
            direction[i + 1] += grad_b
            direction[i] -= (grad_a + grad_b)

    return energy, direction


# Author: Jimmy
def get_ca_energy(xyz, original_xyz, cispro, max_shift):
    energy, _ = get_ca_energy_and_direction(xyz, original_xyz, cispro, max_shift, calc_gradient=False)
    return energy


# Author: Jimmy
def optimize_ca_coordinates(ca, resnames, max_iterations=100, max_shift=3.0, detect_cispro=False, seed=1234):
    original_coords = np.array(ca, dtype=float)
    coords = original_coords.copy()
    num_atoms = len(coords)

    cispro = [False] * num_atoms
    if detect_cispro:
        for i in range(1, num_atoms):
            if resnames[i] == 'PRO':
                distance = np.linalg.norm(coords[i] - coords[i - 1])
                cispro[i] = 2.4 < distance < 3.4

    init_energy = get_ca_energy(coords, original_coords, cispro, max_shift)
    noise_level = 0.5
    prev_norm = 1000.0
    stagnant_count = 0
    rng_seed = seed
    total_iters = 0
    final_norm = 0.0

    for it in range(max_iterations):
        total_iters = it + 1
        curr_energy, step_dir = get_ca_energy_and_direction(
            coords, original_coords, cispro, max_shift
        )

        left = -1.0
        mid = 0.0
        right = 1.0
        left_e = get_ca_energy(coords - step_dir, original_coords, cispro, max_shift)
        right_e = get_ca_energy(coords + step_dir, original_coords, cispro, max_shift)

        for _ in range(max_iterations):
            if curr_energy <= left_e and curr_energy <= right_e:
                break

            left *= 2.0
            right *= 2.0
            left_e = get_ca_energy(coords + left * step_dir, original_coords, cispro, max_shift)
            right_e = get_ca_energy(coords + right * step_dir, original_coords, cispro, max_shift)

        for _ in range(20):
            is_right = (right - mid) > (mid - left)
            if is_right:
                test_pt = 0.5 * (mid + right)
            else:
                test_pt = 0.5 * (left + mid)

            test_e = get_ca_energy(coords + test_pt * step_dir, original_coords, cispro, max_shift)
            if test_e < curr_energy:
                if is_right:
                    left = mid
                else:
                    right = mid
                mid = test_pt
                curr_energy = test_e
            else:
                if is_right:
                    right = test_pt
                else:
                    left = test_pt

            if abs(right - left) <= 1e-6:
                break

        noise_mat = np.zeros((num_atoms, 3))
        for i in range(num_atoms):
            for k in range(3):
                rng_seed = (16807 * rng_seed) % 2147483647
                noise_mat[i, k] = 0.001 * (rng_seed % 1000) - 0.5

        coords += (mid + mid * noise_mat * noise_level) * step_dir
        noise_level *= 0.75
        if noise_level < 0.001:
            noise_level = 0.0

        final_norm = float(np.sqrt(np.sum(step_dir ** 2) / num_atoms))
        if (prev_norm - final_norm) < 0.001:
            stagnant_count += 1
        prev_norm = final_norm

        if stagnant_count >= 3 or final_norm <= 0.01:
            break

    final_energy = get_ca_energy(coords, original_coords, cispro, max_shift)

    info = {
        'initial_energy': float(init_energy),
        'final_energy': float(final_energy),
        'iterations': total_iters,
        'direction_norm': final_norm,
        'method': 'PULCHRA_legacy_energy_and_line_search',
        'seed': seed,
        'rng': 'Park-Miller (matches supplied macOS C runtime)'
    }
    return coords, info


# Task: C6
# Author: Jimmy
def optimize_ca_stage(ca_record, options):
    if not isinstance(options.get('optimize_ca'), bool):
        raise ValueError('optimize_ca must be boolean')

    result = deepcopy(ca_record)
    coords = np.asarray(result['coords'], dtype=float)
    result['coords'] = coords

    if not options['optimize_ca']:
        result['stage_status'] = 'disabled'
        return result

    if any(key[3] != 'CA' for key in result['atom_keys']):
        raise ValueError('C6 expects a CA trace')
    diagnostics = []
    for segment in result['segments']:
        names = [result['residue_names'][i] for i in segment]
        new_coords, info = optimize_ca_coordinates(coords[segment], names)
        coords[segment] = new_coords
        info['segment'] = list(segment)
        diagnostics.append(info)

    result['stage_status'] = 'executed'
    result['diagnostics'] = diagnostics
    return result
