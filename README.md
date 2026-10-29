[README.md](https://github.com/user-attachments/files/33852113/README.md)
# SuperDex

### Differentiable World Models for Dexterous Manipulation

**Learn the dynamics. Imagine the motion. Control the hand.**

SuperDex explores dexterous manipulation through learned dynamics and gradient-based policy optimization. Inspired by [SuperMimic](https://supermimic.github.io/), it brings the idea of learning control through a differentiable world model to the interaction between a multi-fingered robotic hand and an object.

The central question is simple: **can a hand learn to manipulate objects by practicing inside a learned model of its world?**

This README presents the SuperDex system design and methodology.

## Overview

Dexterous manipulation couples articulated motion with changing contacts, friction, and object dynamics. SuperDex approaches this problem with a shared world model and a goal-conditioned controller: the model predicts the consequences of actions, while the controller learns which actions move the object toward its goal.

The design connects three components:

- **A differentiable world model** for predicting hand-object state transitions.
- **A goal-conditioned policy** for coordinating finger motion across target object poses.
- **A simulation feedback loop** for collecting experience and checking whether imagined behavior transfers back to the physical simulation.

## Architecture

| Component | Role |
| --- | --- |
| State representation | Encodes hand joint positions and velocities, object pose and motion, and contact observations. |
| World model | Predicts the next hand-object state from the current state and control action. |
| Control policy | Maps observations and a task goal to bounded actuator commands. |
| Task objective | Combines object pose tracking, object retention, and smooth control. |
| Experience buffer | Stores simulator transitions for world-model updates. |
| Evaluation protocol | Measures task performance in simulation independently of model predictions. |

## Method

### Learn from interaction

The policy interacts with a rigid-body simulator to collect state-action transitions. These transitions supervise the world model, grounding its predictions in simulated hand-object dynamics.

### Optimize through imagination

Starting from sampled states, the policy generates actions inside the learned model. Short imagined trajectories provide a differentiable objective for improving task performance. Model parameters remain fixed during each policy update, while gradients flow through the predicted dynamics to the policy.

This formulation takes gradients through the learned model; it does not require differentiating the simulator itself.

### Close the loop

World-model fitting and policy optimization alternate with new simulator experience. Evaluating the policy back in simulation exposes prediction errors and behavior that exploits inaccuracies in the model.

## Task Formulations

- **In-hand reorientation:** rotate a grasped object toward a target orientation while maintaining control.
- **Grasp stabilization:** preserve a desired object pose under disturbances and changing initial conditions.
- **Trajectory tracking:** follow reference object poses alongside coordinated finger motion.
- **Sequential manipulation:** connect a sequence of target poses into a continuous manipulation task.

The reference setting is a single fixed-base dexterous hand manipulating a rigid object. Object geometry, mass, friction, and observation noise define the conditions for robustness evaluation.

## Evaluation

The evaluation protocol separates control performance from model accuracy:

| Category | Metrics |
| --- | --- |
| Task completion | Success rate, completion time, and object drop rate |
| Tracking quality | Object position and orientation error |
| Control quality | Action magnitude and temporal smoothness |
| Model fidelity | One-step and multi-step state prediction error |
| Training cost | Environment interactions and end-to-end training time |
| Robustness | Performance across held-out physical parameters and initial states |

Task definitions, success thresholds, evaluation seeds, and compute settings belong alongside any reported results so comparisons remain reproducible.

## Acknowledgments

SuperDex draws methodological inspiration from **SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**.

[SuperMimic Project Page](https://supermimic.github.io/)

SuperDex is an independent exploration of these ideas for dexterous manipulation and is not affiliated with the SuperMimic authors. The hand-object architecture and task formulations described here are SuperDex design choices.

## Research implementation

The active workstation's code snapshot is in [`research/`](research/README.md).
It includes the A1 pipeline, collectors, world models, configurations, and
dependency locks. Source revision: `fdc4ce39ef317653e17d225fea9082098940ed13`
(2026-10-29). Datasets, checkpoints, private assets, and credentials are excluded.
The experiment remains in progress; this snapshot does not establish completed
A1 results.
