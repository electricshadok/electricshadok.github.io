---
title: 'Vision-Language-Action Models: A Robotics Curriculum'
date: 2026-09-27
permalink: /posts/2026/09/vla-robotics-curriculum/
tags:
  - robotics
  - vision-language-action
  - imitation learning
  - transformers
  - deep learning
---

The promise of general-purpose robots has always run into the same wall: a robot that can pick up a cup cannot clean a table, and a robot that cleans a table cannot load a dishwasher. **Vision-Language-Action (VLA) models** aim to break that wall by conditioning robot behavior on natural language and grounding high-capacity vision-language representations directly into **low-level motor control**.

If you have followed the rise of large language models and vision-language models (VLMs), VLAs are the logical next step: the same foundation models that describe images and follow instructions are repurposed to *act* — predicting joint angles, gripper commands, and end-effector trajectories from camera observations and text goals.

This post breaks the VLA stack into its three core components — **V** (vision encoding), **L** (language encoding), and **A** (action prediction and multimodal fusion) — then surveys the key papers that define the field. A short detour through **VLMs** covers the foundation layer that makes modern VLAs possible.

---

## The VLA Stack

A VLA model has three interlocking components. Each has its own research literature, and understanding them separately before seeing how they combine is the fastest path to reading the primary papers.

### V — Visual Encoders

A **visual encoder** maps raw pixel observations — RGB images from wrist or head cameras — into a compact **latent representation**: a vector or sequence of vectors that captures semantically meaningful features. In manipulation, the encoder must simultaneously preserve **semantic content** (what objects are) and **spatial structure** (where they are and how they relate).

Most VLAs use encoders **pretrained on internet-scale image or video data** and kept frozen during policy training, treating them as general-purpose perception backbones.

| Encoder | Year | Authors | Architecture | Key Property for Robotics |
|---|:---:|---|---|---|
| **ViT** | 2020 | Dosovitskiy et al. | **Patch tokenization** + multi-head self-attention | First pure-Transformer image encoder; each spatial patch becomes an independent token |
| **CLIP** | 2021 | Radford et al. (OpenAI) | **Contrastive language-image pretraining** | Visual features semantically aligned to natural language — strong zero-shot grounding |
| **R3M** | 2022 | Nair et al. (Meta) | **Time-contrastive** + language-video alignment | Encoder specifically optimized for motor control via human video, not classification |
| **MVP** | 2022 | Radosavovic et al. | **Masked Autoencoder (MAE)** pretraining | Learns dense spatially grounded features via masked patch reconstruction |
| **DINOv2** | 2023 | Oquab et al. (Meta) | Self-supervised ViT + **knowledge distillation** | Strong out-of-the-box dense spatial features; no language supervision required |
| **SigLIP** | 2023 | Zhai et al. (Google) | **Sigmoid contrastive** vision-language pretraining | Outperforms CLIP on robotics benchmarks; backbone of choice in OpenVLA and π0 |

---

### L — Language Encoders

A **language encoder** maps natural language task descriptions — *"pick up the red cup and place it on the tray"* — into **semantic embeddings** that condition the robot policy. The choice of language encoder determines how well the system generalizes to novel, compositional, or abstract instructions.

Early VLAs used lightweight encoders (BERT, T5) purely for **instruction conditioning** — injecting task embeddings into a separate visual policy via **FiLM layers** or cross-attention. Modern VLAs instead treat the language model as the **policy backbone** itself, letting it directly generate action tokens as an extension of its **next-token prediction objective**.

| Encoder | Year | Authors | Architecture | Role in VLA |
|---|:---:|---|---|---|
| **BERT** | 2018 | Devlin et al. (Google) | Bidirectional Transformer **encoder** | Task embedding for early conditioned policies (e.g., CLIPort) |
| **T5** | 2019 | Raffel et al. (Google) | **Encoder-decoder** Transformer | Instruction conditioning via **FiLM** in RT-1 and SayCan affordance models |
| **PaLM** | 2022 | Chowdhery et al. (Google) | **Decoder-only**, 540B parameters | High-capacity semantic reasoning; backbone for SayCan task planning and RT-2 action generation |
| **LLaMA 2** | 2023 | Touvron et al. (Meta) | **Decoder-only**, 7B–70B | Open-source backbone widely adapted for VLA fine-tuning (e.g., OpenVLA variants) |
| **Gemma** | 2024 | Google DeepMind | **Decoder-only**, 2B–7B | Backbone for OpenVLA (Prismatic) and π0 via PaliGemma |

---

### A — Action Prediction & Multimodal Fusion

The action head and fusion mechanism are what transform a vision-language model into a robot policy. Three main families have emerged, each with a different inductive bias.

**Action Prediction Strategies:**

* **Tokenized discrete actions** — continuous joint angles or end-effector deltas are **discretized into fixed bins** (e.g., 256 bins per DoF) and predicted as regular vocabulary tokens. Used by RT-1, RT-2, and Gato. Simple to implement on any LLM, but quantization limits precision for fine-grained control.

* **Diffusion policy heads** — actions are generated by **iteratively denoising** a Gaussian noise vector conditioned on the backbone's latent representation via a **DDPM** process. Used by Octo and Diffusion Policy. Captures **multimodal action distributions** — crucial when multiple valid behaviors exist — but slower at inference.

* **Flow matching** — actions are generated via **continuous normalizing flows**, learning a straight-line ODE path from noise to action distribution. Faster and more training-stable than DDPM. Used by π0 and currently the state-of-the-art approach for dexterous manipulation.

* **Action chunking** — instead of predicting single-step actions, the policy predicts a short-horizon **action sequence** (chunk) at each timestep. Reduces **compounding closed-loop errors** and enables smoother trajectories.

**Multimodal Fusion Strategies:**

* **Linear projection** (LLaVA-style): visual encoder outputs are projected into the LLM's **token embedding space** and prepended as soft visual tokens before the language sequence.
* **Cross-attention** (Flamingo-style): language transformer layers attend over visual tokens at **every block** via dedicated cross-attention, preserving visual context throughout the full forward pass.
* **Token interleaving** (Gato): all modalities — image patches, language tokens, action tokens — are flattened into a single **autoregressive token sequence** processed by one Transformer.

---

## A Note on VLMs: The Foundation Layer

**Vision-Language Models (VLMs)** sit one level below VLAs. They learn rich joint representations of images and language but do not produce robot actions. The key insight of modern VLAs is that a VLM already encodes the semantic understanding needed to *reason about manipulation* — what object to grasp, in what context, following what instruction. The missing piece is grounding that understanding into **motor commands**.

| VLM | Year | Authors | Key Contribution to the VLA Stack |
|---|:---:|---|---|
| **Flamingo** | 2022 | Alayrac et al. (DeepMind) | **Cross-attention** over frozen visual features at each LLM layer; architecture adopted directly by RoboFlamingo |
| **PaLM-E** | 2023 | Driess et al. (Google) | **Embodied language model** — continuous sensor observations (images, robot state) injected as multimodal tokens into a 562B LLM |
| **LLaVA** | 2023 | Liu et al. | **Visual instruction tuning** via a simple linear projection; established the dominant blueprint for VLA pipeline construction |
| **Prismatic** | 2024 | Karamcheti et al. (Stanford) | Systematic study of VLM design choices — encoder selection, fusion, data mix — specifically for robot policy learning; base architecture for OpenVLA |
| **PaliGemma** | 2024 | Google DeepMind | **SigLIP** vision encoder + **Gemma** LLM in a single fine-tunable package; direct backbone for π0 and π0.5 |

---

## 🗺️ The VLA Curriculum Table

| # | Year | Model | V Encoder | L Encoder | Action Head | Summary | Link |
|---|:---:|---|---|---|---|---|:---:|
| **1** | 2021 | **CLIPort** *(Shridhar et al.)* | CLIP ViT | CLIP text encoder | **Transporter** spatial-action network | Uses CLIP's **semantic-spatial** features to condition a pick-and-place planner on language goals — no generative LLM, just contrastive visual grounding matched to instructions. | [arXiv:2109.12098](https://arxiv.org/abs/2109.12098) |
| **2** | 2022 | **SayCan** *(Ahn et al., Google)* | Per-skill value networks | PaLM (540B) | **Affordance value functions** per primitive skill | An LLM generates candidate plans in natural language; each step is scored by a learned **value function** estimating physical feasibility — grounding abstract language reasoning in robot affordances. | [arXiv:2204.01691](https://arxiv.org/abs/2204.01691) |
| **3** | 2022 | **Gato** *(Reed et al., DeepMind)* | ViT patch tokens | Byte-pair tokens (SentencePiece) | **Autoregressive token prediction** | A single **generalist Transformer** processes image patches, language, and actions as a flat **interleaved token sequence** across 600+ tasks. Actions are discretized into vocabulary tokens like any other output. | [arXiv:2205.06175](https://arxiv.org/abs/2205.06175) |
| **4** | 2022 | **RT-1** *(Brohan et al., Google)* | EfficientNet-B3 + **TokenLearner** | T5 (**FiLM** conditioning) | **Tokenized discrete actions** (256 bins/DoF) | A dedicated **Robotics Transformer** trained on 130k real-robot demonstrations. **FiLM layers** inject T5 language embeddings into the visual feature stream; the output head predicts discretized joint commands. | [arXiv:2212.06817](https://arxiv.org/abs/2212.06817) |
| **5** | 2022 | **R3M** *(Nair et al., Meta)* | **Time-contrastive ViT** | CLIP text (alignment signal) | Downstream task-specific policy | A visual pretraining framework using **temporal contrastive learning** on human videos and **video-language alignment** to learn motor-relevant representations — without any robot data. | [arXiv:2203.12601](https://arxiv.org/abs/2203.12601) |
| **6** | 2023 | **RoboFlamingo** *(Li et al.)* | CLIP ViT (Flamingo) | OpenFlamingo LLM | **MLP action decoder** | Fine-tunes **OpenFlamingo** on robot manipulation data using its native **cross-attention over visual tokens** to condition an action MLP. Demonstrates strong **few-shot generalization** from language-conditioned rollouts. | [arXiv:2311.01378](https://arxiv.org/abs/2311.01378) |
| **7** | 2023 | **RT-2** *(Brohan et al., Google)* | PaLI-X ViT (22B) | PaLM-E / PaLI-X | **Co-fine-tuned text tokens** as actions | **Co-fine-tunes** a VLM on robot data *alongside* web data so the model retains language understanding while learning to emit action tokens. Unlocks **emergent chain-of-thought reasoning** — the robot follows novel multi-step instructions unseen during robot training. | [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) |
| **8** | 2024 | **Octo** *(Octo Team, UC Berkeley)* | ViT (frozen) | Small GPT-style Transformer | **Diffusion action head** | An **open-source generalist robot policy** trained on the Open X-Embodiment dataset. Task tokens (language + goal images) are fused via **cross-attention**; actions are decoded by a **DDPM diffusion head** that captures multimodal action distributions. | [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) |
| **9** | 2024 | **OpenVLA** *(Kim et al., Stanford)* | **SigLIP + DINOv2** (dual encoder) | Llama 2 / Gemma (7B) | **Tokenized discrete actions** | An **open-source 7B VLA** built on Prismatic. Uses a **dual vision encoder** — SigLIP for language-aligned semantics, DINOv2 for dense spatial structure — projected into an LLM that outputs discretized robot actions as vocabulary tokens. | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) |
| **10** | 2024 | **π0** *(Black et al., Physical Intelligence)* | SigLIP (via PaliGemma) | Gemma (via PaliGemma) | **Flow matching action expert** | Attaches a **flow matching action expert** to a frozen PaliGemma VLM backbone. The expert generates **continuous action trajectories** by solving a straight-line ODE from noise to actions — achieving state-of-the-art on dexterous bimanual manipulation. | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |
| **11** | 2024 | **GR-2** *(Chang et al.)* | ViT pretrained on video | T5 | **Transformer action decoder** | Pretrains a **video generation model** on large-scale internet video to implicitly learn world dynamics, then fine-tunes it as a robot policy — using **generated future frames** as a planning signal. | [arXiv:2410.06158](https://arxiv.org/abs/2410.06158) |
| **12** | 2024 | **RoboVLMs** *(Li et al.)* | CLIP / SigLIP / DINOv2 | LLaMA / Qwen / InternLM | Various | A **systematic benchmark** evaluating which VLM design choices — encoder selection, LLM scale, fusion strategy, fine-tuning recipe — matter most when adapting VLMs to robot manipulation. Essential reading before building your own VLA. | [arXiv:2406.13287](https://arxiv.org/abs/2406.13287) |
| **13** | 2025 | **π0.5** *(Physical Intelligence)* | SigLIP + video features | Gemma (large) | **Flow matching**, language-conditioned | Extends π0 with **internet-scale video pretraining** and a stronger language backbone, enabling generalization to household manipulation tasks in novel environments without task-specific demonstrations. | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) |
| **14** | 2025 | **GR00T N1** *(NVIDIA)* | ViT (multi-view) | LLM backbone | **Diffusion Transformer (DiT)** action head | NVIDIA's **open foundation robot policy** trained on diverse cross-embodiment data. Uses a **Diffusion Transformer** action head and a multi-view visual encoder, targeting humanoid and dexterous manipulation robots. | [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) |
