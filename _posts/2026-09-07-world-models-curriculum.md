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

Model-Based Reinforcement Learning has transformed over the past decade. What started as recurrent neural nets hallucinating simple race tracks has turned into real-time interactive game engines and predictive models that help physical robots move in the real world.

If you want to understand how world models actually work, reading random papers won't get you very far. You need to follow the story of why the older architectures failed, what ideas fixed them, and how modern techniques handle long-term planning without getting confused.

Here is a 17-paper roadmap broken down into three straightforward acts.

---

## The Three Acts of World Model Evolution

### Act I: The Latent State Foundations (2018–2020)
* **The Problem:** How can an AI predict what happens next without burning compute trying to predict every single pixel on screen?
* **The Breakthrough:** Separate compression from time prediction.
* Ha & Schmidhuber showed you could train an agent entirely inside the "dreams" of a recurrent neural net. Hafner and team added structure with the Recurrent State-Space Model (RSSM) in *PlaNet* and *DreamerV1*, splitting memory into a reliable predictable path and a random path for uncertainty. At the same time, DeepMind's *MuZero* took the opposite approach: stop predicting images entirely and only predict rewards and actions directly.

### Act II: Keeping Things Stable & The Transformer Shift (2021–2023)
* **The Problem:** Continuous numbers caused the model to lose focus over long horizons. Hallucinations quickly compounded into complete nonsense.
* **The Breakthrough:** Discrete chunks, scale tricks, and Transformers.
* *DreamerV2* swapped continuous smooth representations for discrete categories (like picking from a set of concepts), stopping the blur and drift in retro games. *DayDreamer* brought this to real physical robots with no pretraining in simulators. *DreamerV3* used simple math transforms so one set of settings worked everywhere—from basic control tasks to finding diamonds in Minecraft. Meanwhile, *TransDreamer*, *IRIS*, and *Δ-IRIS* swapped out recurrent loops for Transformers, treating visual reality like sentences of tokens.

### Act III: Diffusion and Predicting Concepts, Not Pixels (2024–2026)
* **The Problem:** Turning everything into discrete tokens loses sharp geometry and subtle physics, while generating long sequences of tokens is slow and prone to visual glitches.
* **The Breakthrough:** Image diffusion and non-generative concept prediction.
* *DIAMOND* and *Oasis* introduced diffusion (the tech behind modern image generators) directly into world modeling, making frame-by-frame simulation look crisp and physically believable in real time. Meanwhile, Meta's *V-JEPA* proved that predicting high-level visual features instead of raw pixels leads to much better physical common sense. This brings us to systems like *Dreamer 4* and *JEDI*, which combine feature-based understanding with fast latent diffusion to imagine possible futures quickly and accurately.

---

## 🗺️ The World Models Curriculum Table

| # | Year | Model / Paper | Core Architecture | Plain English Summary | Link | Code |
|---|:---:|---|---|---|:---:|:---:|
| **1** | 2018 | **World Models** *(Ha & Schmidhuber)* | VAE + MDN-RNN + Linear Controller | Compresses frames into small vectors, uses a memory net to predict the future, and trains a basic controller inside the dreams. | [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) | [GitHub](https://github.com/ctallec/world-models) |
| **2** | 2019 | **PlaNet** *(Hafner et al.)* | Recurrent State-Space Model (RSSM) + Planner | Combines deterministic memory with random variables so the model can handle uncertainty while planning directly in latent space. | [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) | [GitHub](https://github.com/google-research/planet) |
| **3** | 2020 | **DreamerV1** *(Hafner et al.)* | RSSM + Actor-Critic | Instead of planning from scratch at every step, it trains a separate policy network using backpropagation through imagined futures. | [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) | [GitHub](https://github.com/danijar/dreamer) |
| **4** | 2020 | **MuZero** *(Schrittwieser et al.)* | Representation + Dynamics + Prediction Nets | Skips image reconstruction completely. Learns only the hidden states necessary to predict rewards, values, and the best move. | [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) | [GitHub](https://github.com/werner-duvaud/muzero-general) |
| **5** | 2021 | **DreamerV2** *(Hafner et al.)* | RSSM with Categorical Latents + Actor-Critic | Switches from smooth continuous values to discrete categorical variables, which stops the agent from getting confused over long rollouts. | [arXiv:2012.09688](https://arxiv.org/abs/2012.09688) | [GitHub](https://github.com/danijar/dreamerv2) |
| **6** | 2022 | **TransDreamer** *(Chen et al.)* | Transformer State-Space Model | Replaces the recurrent neural net memory with attention layers, helping the model remember information across longer horizons. | [arXiv:2202.09481](https://arxiv.org/abs/2202.09481) | [GitHub](https://github.com/yd-c/TransDreamer) |
| **7** | 2022 | **IRIS** *(Micheli et al.)* | Discrete Autoencoder + Autoregressive Transformer | Turns frames into visual tokens (like words) and uses a language-model style Transformer to predict what token comes next. | [arXiv:2209.00588](https://arxiv.org/abs/2209.00588) | [GitHub](https://github.com/eloialonso/iris) |
| **8** | 2022 | **DayDreamer** *(Wu et al.)* | DreamerV2 on Real Hardware | Runs Dreamer directly on physical robots, showing a robot dog can learn to stand and walk in an hour without any simulator. | [arXiv:2206.14176](https://arxiv.org/abs/2206.14176) | [GitHub](https://github.com/danijar/daydreamer) |
| **9** | 2022 | **TD-MPC** *(Hansen et al.)* | Task-Oriented Dynamics + Trajectory Planner | Uses short-horizon planning directly inside a task-focused latent space without rebuilding pixels, making continuous robot control fast. | [arXiv:2203.04955](https://arxiv.org/abs/2203.04955) | [GitHub](https://github.com/nicklashansen/tdmpc) |
| **10** | 2023 | **DreamerV3** *(Hafner et al.)* | RSSM (discrete) + Symlog normalization | Normalizes rewards and prediction values across wildly different scales, allowing one recipe to master Atari, robotics, and Minecraft. | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) | [GitHub](https://github.com/danijar/dreamerv3) |
| **11** | 2024 | **V-JEPA** *(Bardes et al.)* | Vision Transformer + Masked Feature Prediction | Learns how the physical world works by masking parts of a video and predicting missing concepts in feature space, not raw pixels. | [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) | [GitHub](https://github.com/facebookresearch/vjepa) |
| **12** | 2024 | **Δ-IRIS** *(Alonso et al.)* | Autoencoder with Delta Tokens + Transformer | Only predicts tokens that change from frame to frame, making long Transformer rollouts significantly faster and cheaper. | [arXiv:2406.19320](https://arxiv.org/abs/2406.19320) | [GitHub](https://github.com/eloialonso/delta-iris) |
| **13** | 2024 | **TD-MPC2** *(Hansen et al.)* | Scaled Transformer/MLP + Multi-task Embeddings | Scales up model-based planning to handle dozens of different robot arms and bodies using a single shared model. | [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) | [GitHub](https://github.com/nicklashansen/tdmpc2) |
| **14** | 2024 | **DIAMOND** *(Alonso et al.)* | Continuous Diffusion Model + Actor-Critic | Uses diffusion to generate future frames, capturing subtle physics and sharp visual details that discrete tokens blur out. | [arXiv:2405.12399](https://arxiv.org/abs/2405.12399) | [GitHub](https://github.com/eloialonso/diamond) |
| **15** | 2024 | **Oasis** *(Decart & Etched)* | Latent Diffusion Transformer (DiT) | An interactive game engine generated entirely on the fly by a diffusion model running in response to keyboard and mouse inputs. | [Project Oasis](https://oasis.decart.ai) | [GitHub](https://github.com/etched-ai/oasis-model) |
| **16** | 2025 | **Dreamer 4** *(Hafner et al.)* | Block-Causal Transformer + Pretraining | Replaces recurrent cells with a scalable transformer trained on massive video libraries to plan over long, complex tasks. | [arXiv:2509.24527](https://arxiv.org/abs/2509.24527) | [Website / Code](https://danijar.com/project/dreamer4/) |
| **17** | 2026 | **JEDI** *(Brunner et al.)* | JEPA Feature Space + Latent Diffusion | Combines feature-level concept prediction with fast latent diffusion to handle multiple possible futures without pixel blur. | [arXiv:2605.13013](https://arxiv.org/abs/2605.13013) | [arXiv Paper](https://arxiv.org/abs/2605.13013) |
