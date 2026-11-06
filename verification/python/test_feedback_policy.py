from neuroedge.control.feedback_policy import FeedbackProposal, LaggedFeedbackGate
from neuroedge.speech.offset_reference import OffsetTable


def test_strictly_lagged_bounded_and_counted():
    m = OffsetTable()
    gate = LaggedFeedbackGate(max_step_q4=8, min_trials_between_writes=2)
    p = FeedbackProposal(0, 0, 9, 6, 'supervised-fixture')
    assert not gate.try_apply(p, next_trial=0, table=m, safe_boundary=True).accepted
    assert not gate.try_apply(p, next_trial=1, table=m, safe_boundary=False).accepted
    m.begin_bin()
    assert not gate.try_apply(p, next_trial=1, table=m, safe_boundary=True).accepted
    m.end_bin()
    d = gate.try_apply(p, next_trial=1, table=m, safe_boundary=True)
    assert d.accepted and d.new_offset_q4 == 6
    assert m.read_channel(9) == 6 and m.accepted_writes == 1
    assert not gate.try_apply(p, next_trial=2, table=m, safe_boundary=True).accepted
    another = FeedbackProposal(1, 1, 9, 5, 'next-label')
    assert not gate.try_apply(another, next_trial=2, table=m, safe_boundary=True).accepted
    assert gate.try_apply(another, next_trial=3, table=m, safe_boundary=True).accepted
    assert m.accepted_writes == 2 and m.read_channel(9) == 11


def test_future_feedback_and_unbounded_steps_are_rejected():
    m = OffsetTable()
    gate = LaggedFeedbackGate(max_step_q4=8, max_abs_offset_q4=8)
    for p in [
        FeedbackProposal(4, 4, 0, 1, 'future'),
        FeedbackProposal(0, 0, 0, 9, 'oversized'),
        FeedbackProposal(0, 0, 512, 1, 'invalid-index'),
        FeedbackProposal(0, 0, 0, 1, ''),
    ]:
        assert not gate.try_apply(p, next_trial=3, table=m, safe_boundary=True).accepted
    assert gate.try_apply(FeedbackProposal(0, 0, 0, 8, 'valid'),
                          next_trial=3, table=m, safe_boundary=True).accepted
    assert not gate.try_apply(FeedbackProposal(1, 1, 0, 1, 'exceeds-absolute-bound'),
                              next_trial=6, table=m, safe_boundary=True).accepted
    assert m.accepted_writes == 1
