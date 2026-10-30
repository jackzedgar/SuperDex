"""CPU contract checks; no simulator, training, or final-evaluation data."""
import itertools
import unittest
import numpy as np
from a1_contact_phase_reporting import phases, exclusive_phases


class Checks(unittest.TestCase):
    def test_all_contact_sequences_preserve_original_masks_and_partition(self):
        patterns = np.asarray(list(itertools.product([0,1], repeat=6)))
        contacts = np.zeros((len(patterns)*2,6,7))
        contacts[:len(patterns),:,0] = patterns
        contacts[len(patterns):,:,1] = patterns
        contacts[::3,3,6] = 1
        contacts[::2,:,2] = 1
        labels=phases(contacts)
        hand=contacts[:,:,:2].sum(-1)>0; floor=contacts[:,:,2]>0
        onset=(~hand[:,:-1]&hand[:,1:]).any(1); loss=(hand[:,:-1]&~hand[:,1:]).any(1)
        original=dict(free=(~hand).all(1),onset=onset,break_contact=loss,switch=onset|loss,
                      sustained=hand.all(1),slip=(contacts[:,:,6]>0).any(1),hand_involved=hand.any(1),
                      support_only=(~hand).all(1)&floor.any(1),unsupported_free=(~hand).all(1)&(~floor).all(1))
        self.assertEqual(set(labels),set(original))
        for name in labels:np.testing.assert_array_equal(labels[name],original[name])
        exclusive=exclusive_phases(labels)
        np.testing.assert_array_equal(np.stack(list(exclusive.values())).sum(0),np.ones(len(contacts)))

    def test_declared_precedence_and_multilabel_overlap(self):
        contacts=np.zeros((5,3,7))
        contacts[0,:,0]=[1,0,1]  # break and onset, also slip
        contacts[1,:,0]=[0,1,1]  # onset and slip
        contacts[2,:,0]=[1,1,1]  # slip and sustained
        contacts[3,:,0]=[1,1,1]  # sustained only
        contacts[:3,1,6]=1
        labels=phases(contacts); exclusive=exclusive_phases(labels)
        for i,name in enumerate(['break_contact','onset','slip','sustained','free']):
            self.assertTrue(exclusive[name][i]); self.assertEqual(sum(v[i] for v in exclusive.values()),1)
        self.assertTrue(labels['onset'][0] and labels['break_contact'][0] and labels['slip'][0])

    def test_support_only_is_retained_separately(self):
        contacts=np.zeros((2,3,7));contacts[0,:,2]=1
        labels=phases(contacts);exclusive=exclusive_phases(labels)
        np.testing.assert_array_equal(labels['support_only'],[True,False])
        np.testing.assert_array_equal(labels['unsupported_free'],[False,True])
        self.assertTrue(exclusive['free'].all())

    def test_aggregate_contact_semantics_are_not_silently_redefined(self):
        contacts=np.zeros((1,3,7));contacts[0,:,0]=[1,1,0];contacts[0,:,1]=[0,1,1]
        labels=phases(contacts)
        self.assertTrue(labels['sustained'][0]);self.assertFalse(labels['switch'][0])

    def test_nonfinite_contact_data_rejected(self):
        contacts=np.zeros((1,3,7));contacts[0,0,0]=np.nan
        with self.assertRaises(AssertionError):phases(contacts)


if __name__=='__main__':unittest.main(verbosity=2)
