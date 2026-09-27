---
title: 'Mastering World Models: From Simple Hallucinations to Latent Diffusion'
date: 2026-09-07
permalink: /posts/2026/09/world-models-curriculum/
tags:
  - reinforcement learning
  - world models
  - deep learning
  - model-based RL
---

Model-Based Reinforcement Learning has transformed over the past decade. What started as recurrent neural nets producing **inaccurate pixel-level rollouts** — hallucinating race tracks — has turned into real-time interactive neural game engines and predictive models deployed on physical robots.

If you want to understand how world models actually work, reading papers at random won't get you far. You need to follow the story of why older architectures failed, what ideas fixed them, and how modern techniques manage **long-horizon planning** without **compounding rollout errors** — the gradual drift that turns predictions into nonsense.

Here is a 17-paper roadmap broken down into three chronological phases.

---

## The Three Acts of World Model Evolution

### Act I: The Latent State Foundations (2018–2020)
* **The Problem:** How can an agent predict future states without incurring the prohibitive cost of **pixel-level reconstruction** — predicting every pixel on screen?
* **The Breakthrough:** **Decouple observation encoding from temporal dynamics modeling** — compress observations into a compact latent space first, then predict forward in that space.
* Ha & Schmidhuber showed you could train an agent entirely inside the "dreams" of a recurrent neural net. Hafner et al. added structure with the **Recurrent State-Space Model (RSSM)** in *PlaNet* and *DreamerV1*, factoring the latent state into a **deterministic recurrent component** and a **stochastic component** — a reliable memory path and a random path for uncertainty. DeepMind's *MuZero* took the opposite approach: learn a purely **value-equivalent model**, predicting rewards, values, and policy logits without reconstructing observations at all.

### Act II: Stabilizing Latent Representations & The Transformer Shift (2021–2023)
* **The Problem:** **Continuous latent representations** accumulated drift over long rollout horizons — **compounding prediction errors** leading to distributional shift and model collapse.
* **The Breakthrough:** **Categorical latent spaces**, reward normalization, and **attention-based sequence models**.
* *DreamerV2* replaced Gaussian latents with **straight-through categorical representations** — discrete slots instead of smooth continuous values — reducing **posterior collapse** and latent drift in Atari games. *DayDreamer* extended this to real physical robots without **sim-to-real transfer**. *DreamerV3* introduced **symlog normalization** — a simple log-scaling transform — so a single set of hyperparameters generalized across domains, from locomotion to finding diamonds in Minecraft. Meanwhile, *TransDreamer*, *IRIS*, and *Δ-IRIS* replaced **recurrent state-space models (RSSMs)** with **attention-based sequence models**, representing observations as **discrete token sequences** via VQ-based tokenization.

### Act III: Diffusion and Latent Feature Prediction (2024–2026)
* **The Problem:** **VQ tokenization** fails to preserve **high-frequency spatial structure** — sharp geometry and subtle physics — while **autoregressive token generation** is slow and prone to visual artifacts over long sequences.
* **The Breakthrough:** **Denoising diffusion probabilistic models (DDPMs)** for simulation and **non-generative latent feature prediction** as a self-supervised objective.
* *DIAMOND* and *Oasis* applied **latent diffusion** directly to world modeling — enabling **high-fidelity, temporally consistent environment simulation** (crisp, physically believable frame-by-frame rollouts) in real time. Meanwhile, Meta's *V-JEPA* demonstrated that **masked feature prediction** in a frozen embedding space — predicting abstract representations rather than raw pixels — yields stronger **physical inductive bias** and sample efficiency. This culminates in *Dreamer 4* and *JEDI*, which combine **JEPA-style representation learning** with **fast stochastic latent diffusion** for efficient multi-step rollouts across multiple possible futures.

---

## 🗺️ The World Models Curriculum Table

| # | Year | Model / Paper | Core Architecture | Plain English Summary | Link | Code |
|---|:---:|---|---|---|:---:|:---:|
| **1** | 2018 | **World Models** *(Ha & Schmidhuber)* | VAE + MDN-RNN + Linear Controller | Encodes frames into a compact **latent vector** via a VAE, uses an MDN-RNN to predict future latents, and trains a linear controller via **CMA-ES** entirely inside simulated rollouts ("dreams"). | [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) | [GitHub](https://github.com/ctallec/world-models) |
| **2** | 2019 | **PlaNet** *(Hafner et al.)* | Recurrent State-Space Model (RSSM) + Planner | Combines a **deterministic recurrent component** with **stochastic latent variables** so the model handles uncertainty, then plans directly in latent space via **CEM** — no pixel reconstruction needed. | [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) | [GitHub](https://github.com/google-research/planet) |
| **3** | 2020 | **DreamerV1** *(Hafner et al.)* | RSSM + Actor-Critic | Instead of explicit **tree search or CEM planning** at each timestep, trains a separate **actor-critic policy** via **backpropagation through imagined rollouts** in latent space. | [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) | [GitHub](https://github.com/danijar/dreamer) |
| **4** | 2020 | **MuZero** *(Schrittwieser et al.)* | Representation + Dynamics + Prediction Nets | Learns a purely **value-equivalent model** — no observation reconstruction. Predicts rewards, values, and optimal action via **Monte Carlo Tree Search (MCTS)** in a task-oriented latent space. | [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) | [GitHub](https://github.com/werner-duvaud/muzero-general) |
| **5** | 2021 | **DreamerV2** *(Hafner et al.)* | RSSM with Categorical Latents + Actor-Critic | Replaces **Gaussian latents** with **straight-through categorical representations** — discrete slots instead of continuous values — reducing **posterior collapse** and latent drift over long rollouts. | [arXiv:2012.09688](https://arxiv.org/abs/2012.09688) | [GitHub](https://github.com/danijar/dreamerv2) |
| **6** | 2022 | **TransDreamer** *(Chen et al.)* | Transformer State-Space Model | Replaces the RSSM's recurrent cell with **multi-head self-attention**, enabling the model to capture **long-range temporal dependencies** that recurrent networks struggle to maintain. | [arXiv:2202.09481](https://arxiv.org/abs/2202.09481) | [GitHub](https://github.com/yd-c/TransDreamer) |
| **7** | 2022 | **IRIS** *(Micheli et al.)* | Discrete Autoencoder + Autoregressive Transformer | Tokenizes frames into **discrete visual tokens** via a VQ-based autoencoder, then uses an **autoregressive Transformer** to predict the next token in the sequence. | [arXiv:2209.00588](https://arxiv.org/abs/2209.00588) | [GitHub](https://github.com/eloialonso/iris) |
| **8** | 2022 | **DayDreamer** *(Wu et al.)* | DreamerV2 on Real Hardware | Runs DreamerV2 directly on physical robots — no **sim-to-real transfer**. A quadruped learns locomotion from scratch on real-world data in under an hour. | [arXiv:2206.14176](https://arxiv.org/abs/2206.14176) | [GitHub](https://github.com/danijar/daydreamer) |
| **9** | 2022 | **TD-MPC** *(Hansen et al.)* | Task-Oriented Dynamics + Trajectory Planner | Learns a **task-oriented latent dynamics model** and plans short-horizon trajectories via **MPPI** without pixel reconstruction, enabling fast continuous robot control. | [arXiv:2203.04955](https://arxiv.org/abs/2203.04955) | [GitHub](https://github.com/nicklashansen/tdmpc) |
| **10** | 2023 | **DreamerV3** *(Hafner et al.)* | RSSM (discrete) + Symlog normalization | Applies **symlog normalization** — a symmetric log-scaling transform — to reward and value targets, allowing a single set of **hyperparameters** to generalize across Atari, robotics, and Minecraft. | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) | [GitHub](https://github.com/danijar/dreamerv3) |
| **11** | 2024 | **V-JEPA** *(Bardes et al.)* | Vision Transformer + Masked Feature Prediction | Applies **masked feature prediction** — masking regions of a video and predicting their representations in a frozen **embedding space** rather than raw pixels — yielding strong **physical inductive bias**. | [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) | [GitHub](https://github.com/facebookresearch/vjepa) |
| **12** | 2024 | **Δ-IRIS** *(Alonso et al.)* | Autoencoder with Delta Tokens + Transformer | Predicts only **delta tokens** — the subset of tokens that change between frames — drastically reducing the **autoregressive sequence length** and speeding up long rollouts. | [arXiv:2406.19320](https://arxiv.org/abs/2406.19320) | [GitHub](https://github.com/eloialonso/delta-iris) |
| **13** | 2024 | **TD-MPC2** *(Hansen et al.)* | Scaled Transformer/MLP + Multi-task Embeddings | Scales **task-oriented model-based planning** to multi-task settings with a shared **latent dynamics model** that generalizes across dozens of robot morphologies without per-task fine-tuning. | [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) | [GitHub](https://github.com/nicklashansen/tdmpc2) |
| **14** | 2024 | **DIAMOND** *(Alonso et al.)* | Continuous Diffusion Model + Actor-Critic | Applies a **denoising diffusion model** directly to future frame generation, preserving **high-frequency spatial structure** — sharp geometry and physics details — that VQ tokenization cannot faithfully reconstruct. | [arXiv:2405.12399](https://arxiv.org/abs/2405.12399) | [GitHub](https://github.com/eloialonso/diamond) |
| **15** | 2024 | **Oasis** *(Decart & Etched)* | Latent Diffusion Transformer (DiT) | A **neural game engine** driven by a **Latent Diffusion Transformer (DiT)** that generates the next frame **autoregressively at inference time** in response to keyboard and mouse inputs. | [Project Oasis](https://oasis.decart.ai) | [GitHub](https://github.com/etched-ai/oasis-model) |
| **16** | 2025 | **Dreamer 4** *(Hafner et al.)* | Block-Causal Transformer + Pretraining | Replaces the RSSM with a **block-causal Transformer** pretrained on large video datasets, enabling **long-horizon planning** over complex, temporally extended tasks. | [arXiv:2509.24527](https://arxiv.org/abs/2509.24527) | [Website / Code](https://danijar.com/project/dreamer4/) |
| **17** | 2026 | **JEDI** *(Brunner et al.)* | JEPA Feature Space + Latent Diffusion | Combines **JEPA-style masked feature prediction** with **latent diffusion** to efficiently model a distribution over future trajectories — multiple possible futures without pixel-level blur. | [arXiv:2605.13013](https://arxiv.org/abs/2605.13013) | [arXiv Paper](https://arxiv.org/abs/2605.13013) |
