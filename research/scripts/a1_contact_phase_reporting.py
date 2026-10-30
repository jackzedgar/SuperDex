"""Frozen multilabel phase metrics plus a declared exclusive reporting partition.

Exclusive precedence: break_contact > onset > slip > sustained > free.
Events precede steady phases; break precedes onset if both occur in a window.
This supplementary partition never replaces the original G1 switch population.
Contact means any active hand-object contact, as in the registered evaluator.
"""
import numpy as np

EXCLUSIVE_PRECEDENCE = ('break_contact', 'onset', 'slip', 'sustained', 'free')
EXCLUSIVE_RULE = ('Supplementary exclusive partition: break_contact > onset > slip > sustained > free. '
                  'First matching label wins; each window appears exactly once. '
                  'Original multilabel/G1 switch metrics remain the primary registered metrics. '
                  'Onset/break refer to aggregate any-hand contact, not every individual contact-pair exchange.')


def phases(contacts):
    contacts = np.asarray(contacts)
    assert contacts.ndim == 3 and contacts.shape[1] >= 2 and contacts.shape[2] >= 7
    assert np.isfinite(contacts).all()
    hand = contacts[:, :, :2].sum(-1) > 0
    floor = contacts[:, :, 2] > 0
    onset = (~hand[:, :-1] & hand[:, 1:]).any(1)
    loss = (hand[:, :-1] & ~hand[:, 1:]).any(1)
    slip = (contacts[:, :, 6] > 0).any(1)
    return dict(free=(~hand).all(1), onset=onset, break_contact=loss,
                switch=onset | loss, sustained=hand.all(1), slip=slip,
                hand_involved=hand.any(1), support_only=(~hand).all(1) & floor.any(1),
                unsupported_free=(~hand).all(1) & (~floor).all(1))


def exclusive_phases(labels):
    n = len(labels['free']); assigned = np.zeros(n, dtype=bool); result = {}
    for name in EXCLUSIVE_PRECEDENCE:
        label = np.asarray(labels[name]); assert label.shape == (n,) and label.dtype == np.bool_
        result[name] = label & ~assigned
        assigned |= result[name]
    assert assigned.all(), 'Phase partition must cover every window; do not fabricate an unknown regime'
    assert np.all(np.stack(list(result.values())).sum(0) == 1)
    return result
