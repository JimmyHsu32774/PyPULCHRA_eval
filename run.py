"""Reconstruct a protein backbone from a CA PDB file."""

import argparse
from pathlib import Path

from pypulchra.function import read_ca_record, write_backbone_pdb
from pypulchra.library import parse_backbone_library
from pypulchra.reconstruct import run_reconstruction
from pypulchra.stages import keep_backbone, make_stages, skip_candidates, skip_correction

ROOT = Path(__file__).resolve().parent


# Written by Codex.
def run_backbone(input_path, output_path, optimize_ca=False):
    ca = read_ca_record(input_path)
    library = parse_backbone_library()
    stages = make_stages(library, skip_candidates, keep_backbone, skip_correction)
    options = {'optimize_ca': optimize_ca, 'correct_sidechains': False}
    result = run_reconstruction(ca, options, stages)
    if not result['backbone']['atom_keys']:
        raise ValueError('No segment has the five CA atoms needed to rebuild a backbone')
    write_backbone_pdb(output_path, result['backbone'])
    return result


# Written by Codex.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['reconstruct'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--optimize-ca', action='store_true', help='Run C6 before rebuilding the backbone')
    args = parser.parse_args()
    result = run_backbone(args.input, args.output, args.optimize_ca)
    backbone = result['backbone']
    print(f"Wrote {len(backbone['atom_keys'])} backbone atoms to {args.output}")
    excluded = sum(row['status'] == 'excluded' for row in backbone['diagnostics'])
    if excluded:
        print(f"Excluded {excluded} residues in segments shorter than five CA atoms")


if __name__ == '__main__':
    main()
