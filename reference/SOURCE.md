# Backbone tables

The three C lookup-table headers and their license are in [c-headers.zip](c-headers.zip).
Python reads `reference/nco_data.h` directly from the archive. To inspect the
headers separately, run `python -m zipfile -e reference/c-headers.zip .` from
the project root.

Unmodified nco_data.h and LICENSE copied from the locally pinned reference:
https://github.com/euplotes/pulchra/tree/23a3e4afc01ab48fe81e5d4a8d5902352d8f0977

The SHA-256 of `nco_data.h` is `e1a39bb07d5d6aedaedce926432cd9d2b1f6d6f9f74e5105509504397b0dab16`.

README version: 3.07; C internal version string: 3.06.
Original copyright and MIT-style terms are retained in LICENSE.
This copy also retains rot_data_idx.h and rot_data_coords.h from the team repository.
The parser retains float32 table precision before converting to Python floats.

`run_C.py` invokes the user's installed `pulchra` executable. The pinned
upstream revision initializes `_PRESERVE` to 1. For a build that can both
preserve and optimize Cα coordinates, follow the [build instructions](../README.md#build-pulchra-for-the-c-runner).
