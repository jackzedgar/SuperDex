import unittest
import numpy as np
import mujoco
from a1_contact_velocity_projection import contact_system, project_hand_velocity


class ContactVelocityProjectionTests(unittest.TestCase):
    def test_stationary_plane_removes_only_incoming_velocity(self):
        v, r = project_hand_velocity([-2], [[3]], [[1]], [0], [10])
        np.testing.assert_allclose(v, [0], atol=1e-8)
        self.assertEqual(r['status'], 'projected')
        self.assertAlmostEqual(r['weighted_change_energy'], 6)

    def test_moving_object_offset_is_not_discarded(self):
        v, r = project_hand_velocity([1], [[2]], [[-1]], [.5], [10])
        np.testing.assert_allclose(v, [.5], atol=1e-8)
        self.assertFalse(r['fallback_to_original'])

    def test_coupled_inertia_projection(self):
        v, r = project_hand_velocity([1, 0], [[2, 1], [1, 2]], [[-1, 0]], [0], [10, 10])
        np.testing.assert_allclose(v, [0, .5], atol=1e-7)
        self.assertFalse(r['fallback_to_original'])

    def test_infeasible_box_keeps_original_and_records_failure(self):
        v, r = project_hand_velocity([0], [[1]], [[1]], [-2], [1])
        np.testing.assert_array_equal(v, [0])
        self.assertTrue(r['fallback_to_original'])
        self.assertEqual(r['status'], 'infeasible_coordinate_bounds')

    def test_incompatible_constraints_never_drop_a_contact(self):
        v, r = project_hand_velocity([0], [[1]], [[1], [-1]], [-1, -1], [10])
        np.testing.assert_array_equal(v, [0])
        self.assertTrue(r['fallback_to_original'])
        self.assertEqual(r['contacts'], 2)

    def test_no_contact_or_already_separating_are_identity(self):
        v0 = np.array([.5, -.2])
        for a, c in [(np.empty((0, 2)), []), ([[1, 0]], [0])]:
            v, r = project_hand_velocity(v0, np.eye(2), a, c, [1, 1])
            np.testing.assert_array_equal(v, v0)
            self.assertEqual(r['status'], 'identity_feasible')
        np.testing.assert_array_equal(v0, [.5, -.2])

    def test_native_normal_jacobian_matches_distance_derivative(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
          <worldbody><body name="hand"><joint type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".1" mass="1" contype="1" conaffinity="2" condim="1"/>
          </body><body name="object" pos=".19 0 0"><freejoint/>
            <geom type="sphere" size=".1" mass=".1" contype="2" conaffinity="1" condim="1"/>
          </body></worldbody><actuator><motor joint="0"/></actuator></mujoco>'''.replace('joint type="slide"', 'joint name="slide" type="slide"').replace('joint="0"', 'joint="slide"'))
        d = mujoco.MjData(m);d.qvel[0] = 1;mujoco.mj_forward(m, d)
        q, v = d.qpos.copy(), d.qvel.copy()
        a, c, h, ids = contact_system(m, d)
        self.assertEqual(len(ids), 1)
        predicted = float((a @ v[:m.nu] + c)[0])
        dist = []
        for sign in [-1, 1]:
            tmp = mujoco.MjData(m);tmp.qpos[:] = q
            mujoco.mj_integratePos(m, tmp.qpos, v, sign * 1e-7)
            mujoco.mj_forward(m, tmp);dist.append(tmp.contact[0].dist)
        self.assertAlmostEqual(predicted, (dist[1] - dist[0]) / 2e-7, places=7)
        self.assertLess(predicted, 0)
        np.testing.assert_array_equal(d.qpos, q);np.testing.assert_array_equal(d.qvel, v)
        self.assertGreater(float(h[0, 0]), 0)


if __name__ == '__main__':
    unittest.main()
