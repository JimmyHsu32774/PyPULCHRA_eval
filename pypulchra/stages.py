from functools import partial

from .optimize import optimize_ca_stage
from .reconstruct import rebuild_backbone


# Author: Jimmy
def _backbone_stage(library, ca_record, options):
    """Pass only Y7 options to the team's backbone function."""
    keys = ('include_ca', 'library_version', 'source_commit')
    backbone_options = {key: options[key] for key in keys if key in options}
    return rebuild_backbone(ca_record, library, backbone_options)


# Author: Jimmy
def make_stages(backbone_library, candidates, select, correct):
    """Connect C6 and Y7 to the supplied side-chain stages."""
    return {
        'optimize_ca': optimize_ca_stage,
        'backbone': partial(_backbone_stage, backbone_library),
        'candidates': candidates,
        'select': select,
        'correct': correct,
    }


# Author: Jimmy
def skip_candidates(ca, backbone, options):
    return []


# Author: Jimmy
def keep_backbone(backbone, pools, options):
    backbone['selected_candidate_ids'] = []
    backbone['stage_status'] = 'backbone_only'
    return backbone


# Author: Jimmy
def skip_correction(selected, pools, options):
    selected['stage_status'] = 'disabled'
    return selected
