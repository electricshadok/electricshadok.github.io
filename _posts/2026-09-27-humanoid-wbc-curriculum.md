---
title: 'Humanoid Whole-Body Control: A Robotics Curriculum'
date: 2026-09-27
permalink: /posts/2026/09/humanoid-wbc-curriculum/
tags:
  - robotics
  - humanoid
  - whole-body control
  - reinforcement learning
  - motion imitation
  - deep learning
---

Teaching an arm to pick up a cup is hard. Teaching a humanoid robot with 30+ joints to walk across a room, crouch down, and pick up that same cup — while keeping its balance the entire time — is a fundamentally different problem. This is the domain of **Whole-Body Control (WBC)**: the challenge of generating coordinated, physically consistent motion commands across every joint of a humanoid robot simultaneously.

Where VLA models (Vision-Language-Action) focus on *what* to do and *which object* to interact with, WBC focuses on *how the entire body moves* — guaranteeing balance, managing contact forces, and producing trajectories that are physically realizable on hardware running at 100–500 Hz.

This post maps the WBC stack from the ground up: the control hierarchy, motion data and retargeting, learning frameworks, and the key papers that define the field.

---

## Why Humanoid WBC Is Different from Arm Control

| Aspect | Arm Manipulation (VLA scope) | Humanoid Whole-Body Control |
|---|---|---|
| **Degrees of freedom** | 6–7 (arm) + 1–2 (gripper) | 30–57 (legs + torso + arms + hands) |
| **Action space** | End-effector pose or joint deltas | Full-body **joint positions / velocities / torques** |
| **Core constraint** | Grasping precision | **Dynamic balance** — the robot must not fall |
| **Control frequency** | 10–50 Hz (policy) | 100–500 Hz (WBC + low-level PD controllers) |
| **Primary failure mode** | Wrong grasp, wrong object | Loss of balance → hardware fall → damage |
| **Key challenge** | Semantic grounding | **Loco-manipulation** — locomotion and manipulation simultaneously |

---

## The WBC Control Hierarchy

Modern humanoid systems use a **layered control architecture**. Understanding the layers is the prerequisite to reading the papers.

```
┌─────────────────────────────────────────────────────────┐
│  Task / Language Level  (~1–5 Hz)                        │
│  VLM / LLM: "carry the box to the table"                │
└───────────────────────────┬─────────────────────────────┘
                            │ task tokens / goal embedding
┌───────────────────────────▼─────────────────────────────┐
│  Motion Policy Level  (~10–50 Hz)                        │
│  Learned policy: outputs reference joint trajectories   │
│  (motion tracking, flow matching, diffusion)            │
└───────────────────────────┬─────────────────────────────┘
                            │ reference q, q̇
┌───────────────────────────▼─────────────────────────────┐
│  Whole-Body Controller  (~100–500 Hz)                    │
│  Solves QP / MPC: maps reference motion to feasible     │
│  joint torques respecting contact constraints           │
└───────────────────────────┬─────────────────────────────┘
                            │ joint torque commands τ
┌───────────────────────────▼─────────────────────────────┐
│  Actuator / PD Level  (~1 kHz)                          │
│  Low-level servo control on each joint                  │
└─────────────────────────────────────────────────────────┘
```

**Key concepts at each layer:**

* **Motion policy** — learned (via RL or imitation learning) neural network that maps observations (proprioception + optionally vision) to reference joint trajectories. This is where the papers below live.
* **Whole-Body Controller (WBC)** — a real-time **Quadratic Program (QP)** or **Model Predictive Controller (MPC)** that takes reference motions and solves for physically consistent joint torques, respecting contact forces, friction cones, and joint limits.
* **Sim-to-real transfer** — because collecting real hardware data is dangerous and slow, policies are almost always trained in GPU-accelerated physics simulators (Isaac Sim, MuJoCo, Genesis) and then transferred to real hardware via **domain randomization**.

---

## Motion Data & The Retargeting Pipeline

Human motion capture is the primary source of behavior for humanoid policies. The pipeline from raw MoCap to robot training data has three steps.

### Step 1 — Capture: Human Motion Datasets

| Dataset | Year | Scale | Format | Key Property |
|---|:---:|---|---|---|
| **AMASS** | 2019 | 40h+, 300+ subjects | **SMPL** body model | The standard large-scale MoCap corpus; unifies 15+ MoCap collections into a single parameterization |
| **CMU MoCap** | 2003 | 2,500+ clips | BVH | Classic dataset; walking, running, sports, daily activities |
| **HumanML3D** | 2022 | 14,616 motions, 44k descriptions | SMPL + text | **Language-annotated** motion — enables text-conditioned motion generation |
| **Motion-X** | 2023 | 81k sequences | SMPL-X (with hands) | Covers **whole-body** including fine finger motion; better suited for dexterous tasks |
| **LAFAN1** | 2020 | 496k frames | BVH | High-quality transitions; widely used for motion interpolation and retargeting evaluation |

### Step 2 — Retargeting: Human → Robot Morphology

Raw human MoCap cannot be applied directly to a robot. Human and robot morphologies differ in joint limits, link lengths, and mass distribution. **Retargeting** maps SMPL joint angles to robot-specific joint configurations.

Key challenges:
* **Foot sliding** — naive retargeting produces non-physical contact artifacts
* **Self-collision** — human proportions do not match robot link geometry
* **Joint limit violations** — robot servo limits differ significantly from biological ranges

Retargeting tools and methods:
- **Optimization-based retargeting** (classic): minimize kinematic error subject to joint limit and collision constraints
- **GMR (General Motion Retargeting)** — neural retargeting that reduces artifacts via learned priors
- **BeyondMimic** — uses guided diffusion to generate physically plausible retargeted motions
- **HuggingFace LeRobot retargeting** — open-source pipelines for Unitree G1, FourierN1, and others

### Step 3 — Policy Training: RL + Motion Imitation

Once retargeted, motions serve as **reference trajectories** for RL-based motion imitation. The agent learns to track these references on the simulated robot, with rewards penalizing deviation from the reference pose and encouraging physical plausibility.

Key RL frameworks for humanoid motion:
* **AMP (Adversarial Motion Priors)** *(Peng et al., 2021)* — trains a discriminator to distinguish robot motion from reference MoCap; the discriminator reward replaces hand-crafted style rewards
* **PHC (Perpetual Humanoid Controller)** *(Luo et al., 2023)* — tracks the full AMASS dataset with a single policy using a **Mixture of Experts** architecture
* **PULSE** *(Luo et al., 2023)* — a universal motion latent space learned from AMASS, enabling zero-shot composition of novel motions

---

## The Loco-Manipulation Problem

**Loco-manipulation** — moving *and* interacting with the environment simultaneously — is the central open problem in humanoid WBC. It requires the robot to:

1. Maintain **dynamic balance** while walking (locomotion subsystem)
2. Reach, grasp, and manipulate objects (manipulation subsystem)
3. **Coordinate both simultaneously** — e.g., walking while carrying a heavy box changes the center-of-mass and requires constant balance re-computation

Three architectural approaches have emerged:

* **Hierarchical / modular**: separate locomotion and manipulation policies connected by a high-level planner. Simpler to train but struggles at the interface (transitions between walking and manipulation phases).
* **Unified whole-body policy**: a single neural network receives full proprioception and outputs all joint commands. Harder to train but avoids seams between modes — the approach taken by HOVER, OmniH2O, and GR00T N1.
* **Residual learning**: a pre-trained motion tracking **base policy** handles balance and locomotion; a **residual policy** adds object-interaction behavior on top (ResMimic approach). Efficient — reuses existing base policies without retraining.

---

## 🗺️ The Humanoid WBC Curriculum Table

| # | Year | System | Method | Sim | Key Contribution | Link |
|---|:---:|---|---|---|---|:---:|
| **1** | 2021 | **AMP** *(Peng et al., UC Berkeley)* | **Adversarial Motion Priors** + PPO | MuJoCo | Replaces hand-crafted style rewards with a **GAN discriminator** that scores how natural the robot motion looks compared to MoCap reference clips. Foundational for all subsequent motion imitation work. | [arXiv:2104.02180](https://arxiv.org/abs/2104.02180) |
| **2** | 2022 | **ASE** *(Peng et al., UC Berkeley)* | **Adversarial Skill Embeddings** + latent space | MuJoCo | Extends AMP by learning a **structured latent skill space** — the policy can be prompted with a latent code to produce different locomotion styles (run, jump, cartwheel) from the same network. | [arXiv:2205.01906](https://arxiv.org/abs/2205.01906) |
| **3** | 2023 | **PHC** *(Luo et al., CMU)* | **Perpetual Humanoid Controller**, MoE | Isaac Gym | Tracks the entire AMASS dataset (7,000+ clips) with a single policy using a **Mixture of Experts** that activates specialized sub-networks for each motion type. First system to achieve near-universal MoCap coverage. | [arXiv:2305.06456](https://arxiv.org/abs/2305.06456) |
| **4** | 2023 | **HumanPlus** *(Fu et al., Stanford)* | **Human shadowing** + RL in sim | Isaac Gym | Full-stack system: a low-level **shadowing policy** lets the humanoid mirror a human in real-time from an RGB camera; egocentric data is then used to train autonomous manipulation skills. Bridges human demonstration and autonomous execution. | [arXiv:2406.10454](https://arxiv.org/abs/2406.10454) |
| **5** | 2024 | **OmniH2O** *(He et al., CMU + NVIDIA)* | **Kinematic pose as universal interface** + RL | Isaac Gym | Uses **kinematic pose** as the sole interface between teleoperation, retargeting, and autonomous skill learning — one representation covers all modalities. Enables whole-body teleoperation and **vision-based autonomous loco-manipulation** from the same policy. | [arXiv:2406.08858](https://arxiv.org/abs/2406.08858) |
| **6** | 2024 | **HOVER** *(He et al., CMU + NVIDIA)* | **Multi-mode policy distillation** + PPO | Isaac Lab | Consolidates diverse control modes — root velocity for navigation, joint-angle tracking for manipulation — into a **single unified policy** via distillation. Eliminates per-mode retraining and enables seamless mid-task mode transitions. | [arXiv:2410.21229](https://arxiv.org/abs/2410.21229) |
| **7** | 2024 | **Exbody2** *(Ji et al.)* | Whole-body motion imitation + RGB input | Isaac Gym | Extends whole-body motion tracking to use only an **RGB camera** (no MoCap markers) for real-time human motion reference — reducing hardware requirements for teleoperation and demonstration collection. | [arXiv:2412.13196](https://arxiv.org/abs/2412.13196) |
| **8** | 2024 | **GR00T N1** *(NVIDIA)* | **Foundation policy**: VLM + DiT action head | Isaac Lab | NVIDIA's open foundation robot policy — combines a VLM slow loop (~5 Hz) for semantic understanding with a **Diffusion Transformer action head** running at high frequency on proprioception + VLM latents. First foundation model explicitly targeting humanoid whole-body control. | [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) |
| **9** | 2024 | **ResMimic** *(Zhuang et al.)* | **Residual policy** on top of motion tracker | Isaac Lab | Adds object-interaction capability to a frozen base locomotion policy via a **residual network** that outputs joint delta corrections. Efficient: reuses existing WBC base policies without full retraining. | [Project](https://resmimic.github.io) |
| **10** | 2025 | **SONIC** *(Jiang et al., NVIDIA NVLabs)* | **Motion tracking at scale** + domain randomization | Isaac Lab | Proposes a **large-scale motion tracking** framework ("supersizing") using 100k+ retargeted MoCap clips with aggressive **domain randomization**. Achieves natural, contact-rich whole-body behaviors on real hardware with zero-shot sim-to-real transfer. Published in *Science Robotics*. | [arXiv:2511.07820](https://arxiv.org/abs/2511.07820) |
| **11** | 2025 | **WholeBodyVLA** *(various)* | **Unified VLA** for whole-body motor control | Isaac Lab | Integrates egocentric vision and natural language directly into a whole-body motor policy — no explicit trajectory reference needed. Represents the convergence of the VLA and WBC research lines into a single model. | [arXiv](https://arxiv.org/abs/2503.02288) |
| **12** | 2025 | **BeyondMimic** *(various)* | **Guided diffusion** for motion retargeting + RL | Isaac Lab | Uses a **diffusion model** to generate physically plausible retargeted motions from AMASS, then trains a tracking policy on the cleaned trajectories. Significantly reduces the retargeting artifact problem that limits MoCap-based training. | [Project](https://beyondmimic.github.io) |

---

## 📦 Key Datasets & Simulators for Humanoid WBC

### Motion & Demonstration Data

| Dataset / Tool | Year | Scale | Purpose |
|---|:---:|---|---|
| **AMASS** | 2019 | 40h+, SMPL format | Primary MoCap corpus for humanoid motion imitation |
| **HumanML3D** | 2022 | 14k motions + language | Text-conditioned motion; enables language-driven WBC |
| **Motion-X** | 2023 | 81k sequences, SMPL-X | Whole-body inc. finger motion; dexterous loco-manipulation |
| **OmniH2O dataset** | 2024 | ~5k whole-body demos | Teleoperated humanoid demonstrations across 14 tasks |
| **LeRobot humanoid datasets** | 2025 | Growing | Open retargeted AMASS + teleoperation data for Unitree G1 |

### Simulators

| Simulator | Physics Engine | Key Strength |
|---|---|---|
| **Isaac Lab** (NVIDIA) | PhysX 5 | GPU-parallelized; runs 10k+ humanoid envs simultaneously; standard for WBC |
| **MuJoCo** (DeepMind) | Custom | Accurate contact dynamics; widely used in academic RL research |
| **Genesis** | Custom (GPU) | New open-source alternative; exceptionally fast; growing adoption |
| **Webots** / **PyBullet** | Bullet | Open-source; lower fidelity; used for quick prototyping |
