"""Read the PULCHRA backbone fragment tables."""
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np

REFERENCE_ARCHIVE = Path(__file__).resolve().parents[1] / 'reference' / 'c-headers.zip'

# Author: Yitian
def parse_backbone_library(path=REFERENCE_ARCHIVE):
    """Read standard and proline fragments from the reference ZIP or a loose header."""
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as archive:
            text = archive.read('reference/nco_data.h').decode('utf-8', errors='replace')
    else:
        text = path.read_text(encoding='utf-8', errors='replace')

    # Author: Yitian
    def parse_table(name):
        block = re.search(name + r'\[\]\s*=\s*\{(.*?)\n\s*\}\}\};', text, re.S)
        if block is None:
            raise ValueError('table %s not found in %s' % (name, path))
        entries = []
        pattern = r'\{\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\}\s*,\s*\{(.*?)\}\s*\}'
        for match in re.finditer(pattern, block.group(1), re.S):
            bins = [int(value) for value in match.groups()[:3]]
            numbers = [float(value) for value in re.findall(r'-?\d+\.\d+', match.group(4))]
            if len(numbers) != 24:
                raise ValueError('fragment with %d coordinates in %s' % (len(numbers), name))
            # the C table is float; cast so that both sides see identical values
            coords = np.array(numbers, dtype=np.float32).astype(float).reshape(8, 3)
            entries.append({'bins': bins, 'coords': coords})
        return entries

    return {
        'version': 'PULCHRA 3.07 nco_data.h',
        'source_path': str(path.absolute()),
        'standard': parse_table('nco_stat'),
        'proline': parse_table('nco_stat_pro'),
    }
