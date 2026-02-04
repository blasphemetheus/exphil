# Pivotal ML/RL Researchers: A Reading List for ExPhil

> **Purpose:** Curated list of foundational researchers whose work directly informs ExPhil's architecture and training approach. Organized by relevance to game AI and reinforcement learning.

**Last Updated:** 2026-02-03

---

## Quick Reference: Priority Reading

| Priority | Researcher | Key Paper | Why It Matters for ExPhil |
|----------|------------|-----------|---------------------------|
| **Critical** | Richard Sutton | The Bitter Lesson | Scaling philosophy, RL foundations |
| **Critical** | Yann LeCun | JEPA Position Paper | World models, representation learning |
| **Critical** | David Silver | AlphaGo/AlphaZero | Game AI, self-play, MCTS |
| **High** | John Schulman | PPO | Our primary RL algorithm |
| **High** | DeepSeek Team | DeepSeek-R1 | GRPO, reasoning via pure RL |
| **High** | Albert Gu | Mamba | Our primary temporal backbone |
| **High** | Ilya Sutskever | Scaling Laws | Why bigger models work |
| **Medium** | Sergey Levine | Offline RL | Learning from replay data |
| **Medium** | Chelsea Finn | MAML | Fast adaptation, meta-learning |
| **Medium** | Oriol Vinyals | AlphaStar | Real-time game AI, multi-agent |

---

## Tier 1: Foundational Figures ("Must Read")

### Richard Sutton
**Affiliation:** University of Alberta, DeepMind (former), ExperienceFlow.AI (CSO)
**2024 Turing Award Winner** (with Andrew Barto)

**Key Contributions:**
- Temporal Difference Learning (TD) - credit assignment across time
- Policy gradient methods
- The Bitter Lesson - scale beats engineering
- Reward hypothesis - intelligence as reward maximization

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) | 2019 | Core philosophy |
| [Learning to Predict by TD Methods](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf) | 1988 | Foundation of RL |
| [Reward is Enough](https://www.sciencedirect.com/science/article/pii/S0004370221000862) | 2021 | Intelligence from reward |
| [RL: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) | 2018 | Complete RL textbook |

**ExPhil Application:** Our entire training philosophy stems from Sutton. TD learning for credit assignment in long games, sparse rewards over shaped rewards, scaling over engineering.

---

### Yann LeCun
**Affiliation:** Meta AI (VP & Chief AI Scientist), NYU
**2018 Turing Award Winner** (with Hinton and Bengio)

**Key Contributions:**
- Convolutional Neural Networks (LeNet)
- Energy-Based Models
- Joint Embedding Predictive Architecture (JEPA)
- World models for autonomous intelligence

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) | 2022 | JEPA architecture |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 2025 | Video understanding, robotics |
| [Energy-Based Models Tutorial](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf) | 2006 | EBM foundations |
| [VL-JEPA](https://arxiv.org/abs/2512.10942) | 2025 | Vision-language joint embedding |

**ExPhil Application:** JEPA's representation-space prediction is directly applicable. V-JEPA 2's MPC planning could replace reactive policy. H-JEPA maps to Melee's micro/macro decision hierarchy.

---

### David Silver
**Affiliation:** DeepMind (former), Ineffable Intelligence (founder)
**2019 ACM Prize in Computing**

**Key Contributions:**
- AlphaGo - first superhuman Go AI
- AlphaGo Zero - learning without human data
- AlphaZero - general game learning
- MuZero - learned world model + planning

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [Mastering Go with DNNs and Tree Search](https://www.nature.com/articles/nature16961) | 2016 | AlphaGo |
| [Mastering Go without Human Knowledge](https://www.nature.com/articles/nature24270) | 2017 | AlphaGo Zero |
| [AlphaZero](https://arxiv.org/abs/1712.01815) | 2017 | General game learning |
| [MuZero](https://arxiv.org/abs/1911.08265) | 2019 | Learned dynamics + MCTS |

**ExPhil Application:** AlphaZero's self-play framework is our target. MuZero's learned world model + planning is the long-term architecture goal. Silver recently left DeepMind to build "endlessly learning superintelligence via RL" - exactly our philosophy.

---

### Geoffrey Hinton
**Affiliation:** University of Toronto (emeritus), Vector Institute
**2018 Turing Award Winner** - "Godfather of Deep Learning"

**Key Contributions:**
- Backpropagation (with Rumelhart and Williams)
- Boltzmann Machines
- Dropout regularization
- AlexNet (with Krizhevsky and Sutskever)

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) | 1986 | Foundation of training |
| [ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | 2012 | AlexNet |
| [Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html) | 2014 | Regularization |

**ExPhil Application:** Dropout in our networks. Backprop through all our training. His recent warnings about AI risk inform responsible development.

---

## Tier 2: RL Algorithm Inventors

### John Schulman
**Affiliation:** OpenAI (co-founder), leads ChatGPT/RLHF team
**PhD under Pieter Abbeel (Berkeley)**

**Key Contributions:**
- Trust Region Policy Optimization (TRPO)
- Proximal Policy Optimization (PPO)
- RLHF for language models
- ProcGen benchmark

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [TRPO](https://arxiv.org/abs/1502.05477) | 2015 | Stable policy updates |
| [PPO](https://arxiv.org/abs/1707.06347) | 2017 | Our primary RL algorithm |
| [OpenAI Five](https://arxiv.org/abs/1912.06680) | 2019 | Real-time game AI |
| [RLHF Tutorial](https://sites.google.com/view/deep-rl-bootcamp/lectures) | 2017 | Deep RL Bootcamp |

**ExPhil Application:** PPO is our self-play algorithm. TRPO's trust region concept informs stable training. OpenAI Five proves pure RL works for real-time games.

---

### Pieter Abbeel
**Affiliation:** UC Berkeley, Amazon FAR
**Advisor to many influential researchers**

**Key Contributions:**
- TRPO (with Schulman)
- Domain Randomization
- Ring Attention
- Advisor: Schulman (OpenAI), Chelsea Finn, founders of Perplexity, Skild, Physical Intelligence

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [Policy Gradients Tutorial](https://people.eecs.berkeley.edu/~pabbeel/nips-tutorial-policy-optimization-Schulman-Abbeel.pdf) | 2016 | NIPS Tutorial |
| [Apprenticeship Learning via IRL](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf) | 2004 | Imitation learning |
| [Domain Randomization](https://arxiv.org/abs/1703.06907) | 2017 | Sim-to-real transfer |

**ExPhil Application:** Domain randomization for frame delay augmentation. His lab's work on imitation + RL is our exact pipeline.

---

### Sergey Levine
**Affiliation:** UC Berkeley (RAIL Lab)

**Key Contributions:**
- Offline Reinforcement Learning
- Conservative Q-Learning (CQL)
- Robot learning from experience
- Learning + search combination

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [Offline RL: Tutorial, Review](https://arxiv.org/abs/2005.01643) | 2020 | Learning from static data |
| [CQL](https://arxiv.org/abs/2006.04779) | 2020 | Conservative value estimation |
| [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702) | 2015 | Pixel to action |

**ExPhil Application:** Our BC phase is essentially offline RL. CQL's conservative estimation could help with out-of-distribution states. His insight that "learning + search is powerful" aligns with our MPC goals.

---

## Tier 3: Architecture Innovators

### Albert Gu
**Affiliation:** Carnegie Mellon University

**Key Contributions:**
- S4 (Structured State Space Sequence models)
- HiPPO framework for long-range memory
- Mamba (Selective State Spaces)
- Mamba-2 (State Space Duality)

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [S4: Efficiently Modeling Long Sequences](https://arxiv.org/abs/2111.00396) | 2021 | SSM foundations |
| [Mamba](https://arxiv.org/abs/2312.00752) | 2023 | Our primary backbone |
| [Mamba-2](https://arxiv.org/abs/2405.21060) | 2024 | SSM-Attention duality |

**ExPhil Application:** Mamba is our real-time backbone (8.9ms inference). S4's long-range memory solves credit assignment. Mamba-2's connection to attention informs our Jamba hybrid.

---

### Ashish Vaswani & Noam Shazeer
**Affiliation:** Google Brain (former), various startups

**Key Contributions:**
- Transformer architecture ("Attention Is All You Need")
- Multi-head attention
- Scaled dot-product attention
- Position encodings

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Foundation of modern AI |
| [Multi-Query Attention](https://arxiv.org/abs/1911.02150) | 2019 | Efficient attention |

**ExPhil Application:** Our attention backbone. Multi-head attention for capturing different temporal patterns (combos, neutral, reactions). Label smoothing, warmup schedules from Transformer paper.

---

### Kaiming He
**Affiliation:** MIT, Google DeepMind (part-time)
**Most cited 21st century AI paper (ResNet)**

**Key Contributions:**
- ResNet (residual learning)
- Mask R-CNN
- He initialization

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [Deep Residual Learning](https://arxiv.org/abs/1512.03385) | 2015 | Skip connections |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870) | 2017 | Instance segmentation |

**ExPhil Application:** Residual connections in our MLP backbone (`--residual`). His insight that residuals enable very deep networks applies to our temporal models.

---

### Jürgen Schmidhuber
**Affiliation:** KAUST, IDSIA, NNAISENSE (founder)

**Key Contributions:**
- LSTM (with Hochreiter)
- Artificial curiosity (1990, precursor to GANs)
- Meta-learning (1987)
- Compression-based creativity theory

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf) | 1997 | Recurrent architecture |
| [CTC Training](https://www.cs.toronto.edu/~graves/icml_2006.pdf) | 2006 | Sequence labeling |
| [Artificial Curiosity](https://people.idsia.ch/~juergen/interest.html) | 1990 | Intrinsic motivation |

**ExPhil Application:** LSTM/GRU backbones. Curiosity-driven exploration for self-play diversity. His compression theory relates to our embedding design.

---

## Tier 4: Modern Frontier Labs

### DeepSeek Team
**Affiliation:** DeepSeek (China)

**Key Contributions:**
- GRPO (Group Relative Policy Optimization) - simplified PPO
- DeepSeek-R1: reasoning via pure RL (no SFT)
- Multi-Head Latent Attention (MLA)
- DeepSeek-V3: 671B MoE, cost-efficient training

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | Pure RL for reasoning |
| [DeepSeek-V3](https://arxiv.org/abs/2512.02556) | 2025 | Efficient MoE training |

**ExPhil Application:** GRPO could replace PPO (simpler, similar performance). Their finding that "skipping SFT enables novel reasoning" parallels our sparse reward experiments. MLA for memory efficiency.

**Key Insight:** DeepSeek-R1 showed that pure RL without supervised fine-tuning can outperform SFT approaches - this validates our Bitter Lesson alignment.

---

### Dario Amodei & Anthropic
**Affiliation:** Anthropic (CEO), OpenAI (former VP Research)

**Key Contributions:**
- RLHF (co-inventor)
- Constitutional AI
- Scaling laws research
- Responsible Scaling Policy

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | 2022 | Self-supervised alignment |
| [Scaling Laws for Neural LMs](https://arxiv.org/abs/2001.08361) | 2020 | Power laws |

**ExPhil Application:** RLHF concepts for reward modeling. Scaling laws inform our model size decisions. Constitutional AI's self-critique could apply to self-play evaluation.

---

### Mistral AI Team
**Affiliation:** Mistral AI (Paris)

**Key Contributions:**
- Mixtral 8x7B - Sparse MoE
- Efficient inference via routing
- Open-weight frontier models

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | 2024 | Sparse MoE architecture |

**ExPhil Application:** MoE could enable larger models within inference budget. Router networks for conditional computation. Their efficiency innovations apply to real-time constraints.

---

### Ilya Sutskever
**Affiliation:** Safe Superintelligence Inc. (CEO), OpenAI (former co-founder)
**Fellow of the Royal Society**

**Key Contributions:**
- AlexNet (with Hinton, Krizhevsky)
- Sequence-to-sequence learning
- GPT series development
- Scaling hypothesis validation

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215) | 2014 | Encoder-decoder |
| [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | GPT-2 |

**ExPhil Application:** Seq2seq for action sequence modeling. His current focus on RL-based superintelligence (SSI) aligns with game AI as RL testbed.

---

## Tier 5: Game AI Specialists

### Oriol Vinyals
**Affiliation:** DeepMind (Gemini technical lead)

**Key Contributions:**
- AlphaStar (StarCraft II)
- Sequence-to-sequence (with Sutskever)
- Neural machine translation

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [AlphaStar: Grandmaster Level](https://www.nature.com/articles/s41586-019-1724-z) | 2019 | Real-time game AI |
| [Seq2Seq Learning](https://arxiv.org/abs/1409.3215) | 2014 | Sequence modeling |

**ExPhil Application:** AlphaStar is the closest analog to our problem. Their multi-agent league training, population-based self-play, and real-time decision making all apply directly.

---

### Chelsea Finn
**Affiliation:** Stanford (IRIS Lab), Physical Intelligence (co-founder)
**2018 ACM Doctoral Dissertation Award**

**Key Contributions:**
- MAML (Model-Agnostic Meta-Learning)
- One-shot imitation learning
- Robot learning from demonstrations

**Essential Reading:**
| Paper | Year | Relevance |
|-------|------|-----------|
| [MAML](https://arxiv.org/abs/1703.03400) | 2017 | Fast adaptation |
| [One-Shot Visual Imitation](https://arxiv.org/abs/1709.04905) | 2017 | Learning from demos |

**ExPhil Application:** MAML for character adaptation (few-shot learning new characters). One-shot imitation for specific techniques. Her insight that "meta-learning enables quick fine-tuning" applies to our character specialization.

---

## Reading Order Recommendations

### For Understanding Our Current Architecture
1. **PPO** (Schulman) - Our RL algorithm
2. **Mamba** (Gu) - Our temporal backbone
3. **ResNet** (He) - Residual connections
4. **Attention Is All You Need** (Vaswani) - Attention mechanism

### For Future Development
1. **AlphaZero/MuZero** (Silver) - World model + planning
2. **JEPA Position Paper** (LeCun) - Representation learning
3. **DeepSeek-R1** (DeepSeek) - Pure RL reasoning
4. **AlphaStar** (Vinyals) - Real-time game AI

### For Philosophy & Scaling
1. **The Bitter Lesson** (Sutton) - Core philosophy
2. **Reward is Enough** (Silver, Sutton) - Intelligence from reward
3. **Scaling Laws** (Kaplan/Amodei) - Power laws

### For Imitation Learning
1. **Offline RL Tutorial** (Levine) - Learning from data
2. **MAML** (Finn) - Fast adaptation
3. **One-Shot Imitation** (Finn) - Demonstration learning

---

## Citation Statistics (for reference)

| Researcher | Google Scholar Citations | h-index |
|------------|-------------------------|---------|
| Geoffrey Hinton | 800,000+ | 182 |
| Kaiming He | 791,532 | 100+ |
| Ilya Sutskever | 749,428 | 100+ |
| Yoshua Bengio | 700,000+ | 200+ |
| David Silver | 285,834 | 80+ |
| Pieter Abbeel | 239,450 | 100+ |
| Oriol Vinyals | 100,000+ | 100+ |

---

## Summary: Key Lessons for ExPhil

### From Sutton (RL Foundations)
- **Scale beats engineering** - Don't over-engineer embeddings
- **Reward is enough** - Trust sparse rewards with sufficient compute
- **TD learning** - Proper credit assignment over long horizons

### From LeCun (World Models)
- **Predict representations, not pixels** - JEPA approach
- **Hierarchical planning** - Multi-timescale decisions
- **Intrinsic motivation** - Curiosity for exploration

### From Silver (Game AI)
- **Self-play works** - AlphaZero paradigm
- **Learned world models** - MuZero architecture
- **Search + learning** - MCTS with neural guidance

### From DeepSeek (Modern Efficiency)
- **GRPO simplifies PPO** - Easier implementation
- **Pure RL can work** - Skip SFT for reasoning
- **Efficient attention** - MLA for memory

### From Gu (Temporal Modeling)
- **Linear-time sequences** - Mamba for real-time
- **Long-range memory** - S4 for credit assignment
- **SSM-Attention duality** - Hybrid approaches

---

## References

Full paper links included throughout. For papers behind paywalls, arXiv preprints are typically available.

**Lab Websites:**
- [Berkeley RAIL Lab](https://rail.eecs.berkeley.edu/) (Levine)
- [Stanford IRIS Lab](https://ai.stanford.edu/~cbfinn/) (Finn)
- [DeepMind Publications](https://www.deepmind.com/research)
- [OpenAI Research](https://openai.com/research)
- [Meta AI Research](https://ai.meta.com/research/)
- [DeepSeek Publications](https://github.com/deepseek-ai)
