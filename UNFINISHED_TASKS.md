# Unfinished tasks and the 1UBQ demo

The Python command rebuilds the backbone. The 76 residues in the 1UBQ demo yield 305 backbone atoms. The following 16 functions contain only `pass` and have no assigned author.

| Task | Function | File |
|---|---|---|
| Y1 | `match_atom_keys(reference_keys, predicted_keys)` | [function.py](pypulchra/function.py) |
| Y2 | `coordinate_rmsd(reference, predicted)` | [function.py](pypulchra/function.py) |
| Y3 | `dihedral_degrees(p0, p1, p2, p3)` | [function.py](pypulchra/function.py) |
| J1 | `angular_error_deg(a, b, period=360.0)` | [function.py](pypulchra/function.py) |
| J2 | `choose_oracle(candidate_ids, errors)` | [evaluate.py](pypulchra/evaluate.py) |
| J3 | `classify_selection(selected_error, candidate_errors, threshold)` | [evaluate.py](pypulchra/evaluate.py) |
| J4 | `assemble_candidate_record(residue_key, candidates, selected_id, scope)` | [reconstruct.py](pypulchra/reconstruct.py) |
| J7 | `capture_candidate_trace(engine, ca_record, options, scope)` | [reconstruct.py](pypulchra/reconstruct.py) |
| C2 | `pair_overlap_energy(coords_a, coords_b, cutoff)` | [clashes.py](pypulchra/clashes.py) |
| C3 | `icm_sweep(states, unary_costs, pairwise_costs)` | [clashes.py](pypulchra/clashes.py) |
| C4 | `compare_stage_metrics(before, after, atol=1e-10)` | [evaluate.py](pypulchra/evaluate.py) |
| C5 | `select_original_candidate(candidates, context, options)` | [reconstruct.py](pypulchra/reconstruct.py) |
| C7 | `correct_structure_stage(structure, candidates, options)` | [reconstruct.py](pypulchra/reconstruct.py) |
| T2 | `summarize_recovery(labels)` | [evaluate.py](pypulchra/evaluate.py) |
| T3 | `paired_metric_summary(before, after, atol=1e-10)` | [evaluate.py](pypulchra/evaluate.py) |
| T5 | `evaluate_chi1(reference_atoms, predicted_atoms, definitions, angle_fn, error_fn, tolerance=40.0)` | [evaluate.py](pypulchra/evaluate.py) |

The original task list also includes T1 `perturb_ca`, T4 `build_experiment_plan` and T7 `run_experiment_batch`. These functions and `experiment.py` are absent from this copy.

## Connecting side-chain stages to I1

I1 calls CA optimization, backbone reconstruction, candidate generation, selection and correction in that order. The current CLI runs a backbone-only example and writes `result['backbone']`.

1. Supply `candidates(ca, backbone, options)` to `make_stages`. Use J5/J6 to build one pool per residue. A pool provides either `candidate_ids` or `candidates` with an `id` field. Leave `selected_id` unset until selection.
2. Supply `select(backbone, pools, options)`. Return the structure and a `selected_candidate_ids` list in pool order. I1 records each choice in its pool.
3. Supply `correct(selected, pools, options)`. To export side chains, the caller must write all atoms from `result['corrected']` with their chain, residue and atom identities. Evaluation and experiment functions need their own callers.

The integration checks exercised C6, Y7, J5 and J6 with a test pool and selection and correction callbacks. J5/J6 still have notes about input handling and whether the candidate includes CA. Check that convention when connecting C5.

## Running the demo

Use the [README](README.md) for environment setup, commands for both implementations, CA optimization and the single-file example. The Python CLI takes a PDB file and an output filename. The C runner takes input and output directories. To run both at the same time, use two terminal tabs with the same working directory and environment.

Run `python -m pytest -q tests` to check the project. The integration test reads the PDB in `data/demo/Input/`. The Y7 test checks backbone coordinates against a [recorded reference structure](tests/fixtures/y7_1ubq_c_backbone.json), including checks for chain breaks, proline and terminal atoms.

## Python and C outputs for 1UBQ

The [Python run](data/demo/output/1ubq_backbone.pdb) with default settings and the [C run](data/demo/output/C/y7_1ubq_ca.pdb) with `-c` both preserve the input CA coordinates. C writes a blank chain field. For this single-chain input it corresponds to chain A. Atoms were matched by chain, residue number, insertion code, residue name and atom name, then compared in the input coordinate frame.

| Measure | Result |
|---|---:|
| Python backbone atoms | 305 |
| C atoms | 602, including 297 side-chain atoms |
| Matched backbone atoms | 305 of 305 |
| RMSD for 76 CA atoms | 0.000000 Å |
| RMSD for all matched backbone atoms | 0.00119 Å |
| Largest atom displacement | 0.005099 Å at GLN 2 O, chain A |

## Comparing C6 with the C optimizer

The six single-chain inputs were 1UBQ, 108L_A, 109L_A, 10CW_A, 6FSX_B and 2EAB_A. With optimization enabled, both implementations used the same number of iterations on each input (5 to 9). All 1,763 CA coordinates matched at the output PDB precision of 0.001 Å. With optimization disabled, both preserved the input CA coordinates.

The six-protein benchmark used an instrumented C build with coordinate preservation disabled and the `-sv` flags for backbone-only output and an iteration log. To run the optimizer with your own `pulchra` executable, follow the [C build instructions](README.md#build-pulchra-for-the-c-runner). The runner passes `-c` by default and omits it for `--optimize-ca`.

| 1UBQ measure | C6 off | C6 on |
|---|---:|---:|
| Python CA RMS displacement from input | 0.0000 Å | 0.0178 Å |
| Python vs. C CA RMSD | 0.0000 Å | 0.0000 Å |
| Python backbone RMSD from the experimental backbone | 0.4533 Å | 0.4232 Å |

RMSDs use the output PDB coordinates in the original frame. Optimization took six iterations and reduced the 1UBQ energy from 0.37002 to 0.00002. [Python optimized backbone](data/demo/output/1ubq_backbone_optimized.pdb) · [C optimized backbone](data/demo/output/C/1ubq_backbone_optimized.pdb)

Known limitation: exactly coincident consecutive CA atoms give the angle calculation a zero-length bond. The Python optimizer can return NaN coordinates with status `executed`. A scan of 10,000 PDB files in `data/Input/` found no such pair. The optimization comparison used only the six inputs listed above.
