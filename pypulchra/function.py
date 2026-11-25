"""PDB, coordinate and CA record functions."""

from pathlib import Path
import numpy as np

# Task: Y1
# Author: Yitian Xiao (Y1)
def match_atom_keys(reference_keys, predicted_keys):
    """Return the row indices of atoms that appear in both structures."""
    reference_keys = list(reference_keys)
    predicted_keys = list(predicted_keys)

    # a repeated atom would make the correspondence ambiguous
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError('duplicate atom key in reference')
    if len(set(predicted_keys)) != len(predicted_keys):
        raise ValueError('duplicate atom key in prediction')

    # map each predicted atom to its row so the lookup stays O(1)
    predicted_rows = {}
    for row, key in enumerate(predicted_keys):
        predicted_rows[key] = row

    ref_match, pred_match = [], []
    for row, key in enumerate(reference_keys):
        if key in predicted_rows:
            ref_match.append(row)
            pred_match.append(predicted_rows[key])
    return ref_match, pred_match


# Task: Y2
# Author: Yitian Xiao (Y2)
def coordinate_rmsd(reference, predicted):
    """RMSD over corresponding atoms, without any superposition."""
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(predicted, dtype=float)

    if ref.ndim != 2 or ref.shape[1] != 3 or len(ref) == 0:
        raise ValueError('reference must be a nonempty (N, 3) array')
    if pred.shape != ref.shape:
        raise ValueError('reference and prediction must have the same shape')
    if not np.isfinite(ref).all() or not np.isfinite(pred).all():
        raise ValueError('coordinates must be finite')

    diff = pred - ref
    # squaring 1e200 would overflow, so divide by the largest displacement first
    scale = float(np.abs(diff).max())
    if scale == 0.0:
        return 0.0
    diff = diff / scale
    return scale * float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


# Task: Y3
# Author: Yitian Xiao (Y3)
def dihedral_degrees(p0, p1, p2, p3):
    """Signed dihedral angle in degrees, following the b0/b1/b2 convention."""
    points = np.asarray([p0, p1, p2, p3], dtype=float)
    if points.shape != (4, 3):
        raise ValueError('need four 3D points')
    if not np.isfinite(points).all():
        raise ValueError('points must be finite')

    b0 = points[0] - points[1]
    b1 = points[2] - points[1]
    b2 = points[3] - points[2]

    length = np.linalg.norm(b1)
    if length < 1e-12:
        raise ValueError('central bond has zero length')
    axis = b1 / length

    v = b0 - np.dot(b0, axis) * axis
    w = b2 - np.dot(b2, axis) * axis
    if np.linalg.norm(v) < 1e-12 or np.linalg.norm(w) < 1e-12:
        raise ValueError('a projected vector has zero length')

    angle = np.degrees(np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w)))
    # arctan2 returns (-180, 180]; shift the +180 end so the range is [-180, 180)
    if angle >= 180.0:
        return float(angle - 360.0)
    return float(angle)


# Task: Y4
# Author: Yitian
def parse_pdb_atoms(lines):
    """Parse the restricted Y4 ATOM format, preserving accepted atom order."""
    records = []
    seen = set()
    for number, line in enumerate(lines, start=1):
        if not isinstance(line, str):
            raise ValueError(f'Line {number}: expected text')
        record_type = line[:6].strip()
        if record_type == 'ENDMDL':
            break
        if record_type != 'ATOM':
            continue
        if len(line.rstrip('\r\n')) < 17:
            raise ValueError(f'Line {number}: truncated ATOM record')
        if line[16] not in (' ', 'A'):
            continue
        if len(line.rstrip('\r\n')) < 54:
            raise ValueError(f'Line {number}: missing fixed-width fields')
        atom_name = line[12:16].strip()
        if not atom_name:
            raise ValueError(f'Line {number}: missing atom name')
        try:
            residue_number = int(line[22:26])
            xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
        except ValueError as exc:
            raise ValueError(f'Line {number}: invalid number field') from exc
        if not np.isfinite(xyz).all():
            raise ValueError(f'Line {number}: nonfinite coordinate')
        key = (line[21].strip(), residue_number, line[26].strip(), atom_name)
        if key in seen:
            raise ValueError(f'Line {number}: duplicate accepted atom {key}')
        seen.add(key)
        records.append({'key': key, 'xyz': xyz})
    return records


# Task: Y5
# Author: Yitian
def fit_rigid_transform(moving, reference):
    """Least-squares rotation R and translation t so that moving @ R + t matches reference."""
    moving = np.asarray(moving, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if moving.ndim != 2 or moving.shape != reference.shape or moving.shape[1] != 3:
        raise ValueError('need two matching (N, 3) arrays')
    if len(moving) < 3:
        raise ValueError('need at least three points')
    if not np.isfinite(moving).all() or not np.isfinite(reference).all():
        raise ValueError('coordinates must be finite')

    moving_mean = moving.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    centred_moving = moving - moving_mean
    centred_reference = reference - reference_mean
    if (np.linalg.matrix_rank(centred_moving) < 2
            or np.linalg.matrix_rank(centred_reference) < 2):
        raise ValueError('points must span at least two dimensions')

    left, _, right = np.linalg.svd(centred_moving.T @ centred_reference)
    # a negative determinant would turn the data inside out; multiply one axis by -1
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(left @ right))])
    rotation = left @ flip @ right
    translation = reference_mean - moving_mean @ rotation
    return rotation, translation


# Task: Y6
# Author: Yitian
def make_local_frame(origin, forward, plane_point):
    """Right-handed frame; global coordinates are local @ axes.T + origin."""
    origin = np.asarray(origin, dtype=float)
    forward = np.asarray(forward, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float)
    for point in (origin, forward, plane_point):
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('expected three finite 3-vectors')

    e1 = forward - origin
    length = np.linalg.norm(e1)
    if length < 1e-12:
        raise ValueError('forward point must differ from origin')
    e1 = e1 / length

    e2 = plane_point - origin
    e2 = e2 - np.dot(e2, e1) * e1
    length = np.linalg.norm(e2)
    if length < 1e-12:
        raise ValueError('the three points must not be collinear')
    e2 = e2 / length

    e3 = np.cross(e1, e2)
    return origin.copy(), np.column_stack((e1, e2, e3))


# Author: Jimmy
def read_ca_record(path):
    """Read the first model's CA atoms and keep segment boundaries for C6 and Y7."""
    keys, names, coords, segments = [], [], [], []
    seen = set()
    previous = None
    with Path(path).open() as handle:
        for line in handle:
            record_type = line[:6].strip()
            if record_type == 'ENDMDL':
                break
            if record_type == 'TER':
                previous = None
                continue
            if record_type != 'ATOM' or line[12:16].strip() != 'CA':
                continue
            atoms = parse_pdb_atoms([line])
            if not atoms:
                continue
            key = atoms[0]['key']
            if key in seen:
                raise ValueError(f'Duplicate CA atom: {key}')
            if previous is None or key[0] != previous[0] or key[1] - previous[1] not in (0, 1):
                segments.append([])
            segments[-1].append(len(keys))
            keys.append(key)
            names.append(line[17:20].strip())
            coords.append(atoms[0]['xyz'])
            seen.add(key)
            previous = key

    if not keys:
        raise ValueError(f'No CA atoms in {path}')
    return {'atom_keys': keys, 'residue_names': names, 'coords': np.array(coords), 'segments': segments}


# Author: Jimmy
def write_backbone_pdb(path, backbone):
    """Write the atom order and chain breaks returned by the team's Y7 function."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for segment in backbone['segments']:
            for i in segment:
                chain, number, insertion, atom = backbone['atom_keys'][i]
                name = backbone['residue_names'][i]
                x, y, z = backbone['coords'][i]
                handle.write(
                    f'ATOM  {i + 1:5d}  {atom:<3} {name:>3} {chain:1}{number:4d}{insertion:1}   '
                    f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom[0]:>2}\n'
                )
            handle.write('TER\n')
        handle.write('END\n')


# Task: J1
def angular_error_deg(a, b, period=360.0):
    pass
