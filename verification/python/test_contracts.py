from neuroedge.contracts import normalize_contract


def test_version_free_contract_and_old_local_profile_compatibility():
    tag = 'neuroedge-train-only-power2-activation-calibration'
    assert normalize_contract(tag) == tag
    assert normalize_contract('neuroedge-' + 'r' + '1.8-' + 'train-only-power2-activation-calibration') == tag
    assert normalize_contract('NEUROEDGE-' + 'R22-' + 'INT8-excursion-activity') == 'neuroedge-int8-excursion-activity'
    assert normalize_contract('R22-' + 'unlabeled-neural-excursion-activity') == 'neuroedge-unlabeled-neural-excursion-activity'
    assert normalize_contract(None) == ''
