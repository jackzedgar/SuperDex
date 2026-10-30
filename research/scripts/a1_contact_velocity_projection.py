"""CPU prototype for revision3. No integration, policy, reference or model edits.

Project hand velocity in the native hand inertia metric while keeping object
velocity fixed. This only enforces instantaneous non-approach at currently
active hand/object contacts; it does NOT guarantee tracking or dynamic viability.
"""
import numpy as np
import mujoco
from scipy.optimize import minimize, LinearConstraint, Bounds

VELOCITY_TOLERANCE = 1e-7


def contact_system(model, data):
    """Input data must already have mj_forward at the intended qpos/qvel."""
    n = model.nu
    assert model.nv == n + 6 and model.nq == n + 7
    assert np.array_equal(model.jnt_dofadr[:n], np.arange(n))
    ob = model.body('object').id
    rows, identities = [], []
    for i in range(data.ncon):
        c = data.contact[i]
        g0, g1 = map(int, c.geom)
        if min(g0, g1) < 0 or c.efc_address < 0 or c.dist > c.includemargin:
            continue
        bodies = [int(model.geom_bodyid[g]) for g in (g0, g1)]
        objects = [model.body_rootid[b] == ob for b in bodies]
        hands = [model.geom_contype[g] == 1 for g in (g0, g1)]
        if not ((objects[0] and hands[1]) or (objects[1] and hands[0])):
            continue
        jacobians = []
        for body in bodies:
            jp = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jp, None, c.pos, body)
            jacobians.append(jp)
        # MuJoCo contact normal points from geom[0] to geom[1].
        row = c.frame.reshape(3, 3)[0] @ (jacobians[1] - jacobians[0])
        rows.append(row)
        identities.append(dict(geom0=g0, geom1=g1, distance=float(c.dist),
                               position=c.pos.copy().tolist(), normal=c.frame[:3].copy().tolist()))
    full_mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full_mass)
    jac = np.asarray(rows, dtype=float).reshape(-1, model.nv)
    return jac[:, :n], jac[:, n:] @ data.qvel[n:], full_mass[:n, :n].copy(), identities


def project_hand_velocity(v, metric, jac, offset, absolute_bounds):
    """Bounded convex QP; failure returns ORIGINAL v with an explicit status.

    No task or phase is removed/retried. Bounds are prespecified per-coordinate
    maxima over the original nonterminal reference, not tuned from outcomes.
    """
    v = np.asarray(v, dtype=float).copy()
    metric = np.asarray(metric, dtype=float)
    jac = np.asarray(jac, dtype=float).reshape(-1, len(v))
    offset = np.asarray(offset, dtype=float)
    bound = np.asarray(absolute_bounds, dtype=float)
    assert metric.shape == (len(v), len(v)) and offset.shape == (len(jac),)
    assert bound.shape == v.shape and np.all(bound >= 0)
    assert all(np.isfinite(x).all() for x in (v, metric, jac, offset, bound))
    assert np.all(abs(v) <= bound + 1e-12)
    assert np.allclose(metric, metric.T, rtol=1e-12, atol=1e-14)
    np.linalg.cholesky(metric)
    before = jac @ v + offset
    info = dict(contacts=len(jac), minimum_before=float(before.min()) if len(before) else None,
                optimizer_iterations=0, candidate_constraint_violation=0.0,
                fallback_to_original=False, weighted_change_energy=0.0)
    if not len(jac) or before.min() >= -VELOCITY_TOLERANCE:
        info.update(status='identity_feasible', minimum_after=info['minimum_before'])
        return v, info
    # An individual inequality that cannot be met even at its box maximum
    # proves infeasibility. Joint infeasibility is handled by the QP status.
    if np.any(abs(jac) @ bound + offset < -VELOCITY_TOLERANCE):
        info.update(status='infeasible_coordinate_bounds', fallback_to_original=True,
                    minimum_after=info['minimum_before'])
        return v, info
    scale = np.sqrt(np.diag(metric))
    hessian = metric / scale[:, None] / scale[None, :]
    constraint = LinearConstraint(jac / scale[None, :], -before, np.inf)
    result = minimize(lambda z: .5 * z @ hessian @ z, np.zeros(len(v)),
                      jac=lambda z: hessian @ z, method='SLSQP',
                      constraints=[constraint],
                      bounds=Bounds(scale * (-bound - v), scale * (bound - v)),
                      options=dict(ftol=1e-12, maxiter=200))
    candidate = v + result.x / scale
    after = jac @ candidate + offset
    violation = max(0.0, float(-after.min()), float(np.max(abs(candidate) - bound)))
    info.update(optimizer_iterations=int(getattr(result, 'nit', 0)),
                optimizer_message=str(result.message),
                candidate_constraint_violation=violation)
    if not result.success or not np.isfinite(candidate).all() or violation > VELOCITY_TOLERANCE:
        info.update(status='optimizer_or_constraint_failure', fallback_to_original=True,
                    minimum_after=info['minimum_before'])
        return v, info
    delta = candidate - v
    info.update(status='projected', minimum_after=float(after.min()),
                weighted_change_energy=float(.5 * delta @ metric @ delta))
    return candidate, info
