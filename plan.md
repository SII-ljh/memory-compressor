# QCPC：Query-Conditioned Perceiver Compressor

> 基于 Perceiver IO 的长文本压缩方案，融合解耦位置增强（文本侧 RoPE + Latent 侧可学习 Slot PE）与零初始化 Prompt 引导

---

## 1. 设计目标

将长度为 $N$ 的长文本压缩为 $M$ 个 Memory Token（$M \ll N$），作为 Soft Prompt 送入冻结的 LLM Decoder 完成下游生成任务。全程**不经过任何大模型的 Self-Attention 层处理长文本**，计算量从 $O(N^2)$ 降至 $O(M \cdot N)$。

---

## 2. 可配置开关

整个架构通过两个布尔开关控制四种工作模式：

| `use_decoupled_rope` | `use_prompt_bias` | 模式描述 |
|:---:|:---:|:---|
| ✗ | ✗ | **Baseline**：原始 Perceiver IO（可学习位置编码）+ 纯可学习 Latent |
| ✗ | ✓ | 原始 Perceiver IO + Prompt 零初始化偏置 |
| ✓ | ✗ | 解耦位置增强（文本 RoPE + Slot PE）+ 纯可学习 Latent |
| ✓ | ✓ | **Full Model**：解耦位置增强 + Prompt 零初始化偏置 |

---

## 3. 模型架构（五阶段流水线）

### 3.1 Embedding 层

直接复用 Qwen3 的 Embedding Lookup Table（冻结），对长文本 $X$ 和提示词 $P$ 分别提取词向量：

$$E_X = \text{Embed}(X) \in \mathbb{R}^{N \times D}, \quad E_P = \text{Embed}(P) \in \mathbb{R}^{L \times D}$$

不经过任何 Transformer 层，复杂度 $O(N)$。

### 3.2 Latent Array 初始化

**基础部分**：维护一个可学习参数矩阵 $Z_{\text{base}} \in \mathbb{R}^{M \times D}$，采用**截断高斯分布**（Truncated Gaussian）初始化，scale = 0.02，截断范围 $[-2\sigma, 2\sigma]$。在整个训练过程中持续更新。

**Prompt Bias 部分**（仅 `use_prompt_bias=True` 时激活）：

1. 对 $E_P$ 沿序列维度做 Mean Pooling，得到全局提示向量 $\bar{e}_P \in \mathbb{R}^D$。
2. 送入 QueryMapper（两层 MLP：$D \to D_{\text{mid}} \xrightarrow{\text{GELU}} D_{\text{mid}} \to M \times D$），reshape 为 $B \in \mathbb{R}^{M \times D}$。
3. 引入标量门控 $\alpha$（可学习标量）。
4. **零初始化约束**：QueryMapper 第二层的权重和偏置初始化为 0，$\alpha$ 初始化为 0。这保证训练起始时 $B \equiv 0$。

最终 Latent Array：

$$Z = \begin{cases} Z_{\text{base}} & \text{if } \texttt{use\_prompt\_bias} = \text{False} \\ Z_{\text{base}} + \alpha \cdot B & \text{if } \texttt{use\_prompt\_bias} = \text{True} \end{cases}$$

零初始化的意义：阶段二微调开始时，模型行为与阶段一完全一致，不因新增通路破坏已学到的压缩能力。Prompt 信号从零渐进式注入，类似 ReZero / ControlNet 范式。

### 3.3 Read 阶段（Cross-Attention：Latent ← 长文本）

Latent $Z$（长度 $M$）作为 Query，长文本向量 $E_X$（长度 $N$）作为 Key/Value。

#### 模式 A：可学习位置编码（`use_decoupled_rope=False`，原始 Perceiver IO）

沿用原始 Perceiver IO 的做法，位置信息通过**可学习位置编码（Learnable Positional Embedding）**在输入端一次性注入：

- 文本侧：$E_X \leftarrow E_X + \text{PE}_{\text{text}}$，其中 $\text{PE}_{\text{text}} \in \mathbb{R}^{N_{\max} \times D}$ 为可学习参数，按实际长度截取前 $N$ 行。
- Latent 侧：$Z \leftarrow Z + \text{PE}_{\text{latent}}$，其中 $\text{PE}_{\text{latent}} \in \mathbb{R}^{M \times D}$ 为可学习参数。

注入后，后续所有 Cross-Attention 和 Self-Attention 层均使用**不含任何位置编码的标准多头注意力**：

$$Q = Z W^Q, \quad K = E_X W^K, \quad V = E_X W^V$$

标准缩放点积注意力，复杂度 $O(M \cdot N \cdot d)$。位置信息随层数加深逐渐稀释，这是该模式的已知局限。

#### 模式 B：解耦位置增强（`use_decoupled_rope=True`）

不使用可学习位置编码，改为在 **Read 阶段的每一层 Cross-Attention 中**通过解耦的位置通道持续注入位置信息。借鉴 DeepSeek MLA 中内容与位置解耦的思想，将 Q 和 K 各拆分为**内容通道**和**位置通道**，但两侧采用不同的位置编码策略：

**KV 侧**（对长文本 $E_X$ 的第 $i$ 个 token）：

- 内容：$K_i^C = E_X^{(i)} W^K$，维度 $d_h$，**不施加 RoPE**。
- 位置：$K_i^R = \text{RoPE}(E_X^{(i)} W^{KR}, \;\text{pos}=i)$，维度 $d_R$，独立投影后施加 RoPE，编码文本中的**真实绝对位置**。
- 拼接：$K_i = [K_i^C \;;\; K_i^R]$。

**Query 侧**（对 Latent $Z$ 的第 $j$ 个 slot）：

- 内容：$Q_j^C = Z_j W^{QC}$，维度 $d_h$，**不施加 RoPE**。
- 位置：$Q_j^R = P_j^{\text{slot}}$，维度 $d_R$，其中 $P^{\text{slot}} \in \mathbb{R}^{M \times d_R}$ 是**可学习 Slot Position Embedding**，截断高斯初始化（scale = 0.02），**不做任何旋转变换**。每个注意力头共享同一组 $P^{\text{slot}}$。
- 拼接：$Q_j = [Q_j^C \;;\; Q_j^R]$。

> **设计动机**：Latent slot 不对应真实序列位置，给它们施加 RoPE 的伪位置 $0, 1, \dots, M-1$ 是人为引入的无意义的顺序偏置。改用可学习向量后，每个 slot 自由学到"倾向关注长文本哪个区域"的位置偏好，而文本侧的 RoPE 则忠实编码 token 的真实绝对位置。

**注意力得分**自然分解为内容匹配 + 位置偏好的叠加：

$$\text{score}_{j,i} = \frac{Q_j^C {(K_i^C)}^\top + P_j^{\text{slot}} {(K_i^R)}^\top}{\sqrt{d_h + d_R}}$$

**V 投影不受影响**，仍为标准的 $V = E_X W^V$。

**收益**（相比模式 A 的可学习位置编码）：

- **文本侧保留真实位置编码**：文本 K 侧通过 RoPE 在每层 Cross-Attention 中持续注入真实绝对位置，位置感知不会随层数加深而稀释。
- **内容与位置不互相干扰**：内容通道纯粹基于语义匹配，不受旋转变换扭曲；位置通道独立编码位置偏好。
- **Latent 获得自由的区域偏好能力**：每个 slot 通过可学习的 $P^{\text{slot}}$ 向量自由学到"倾向关注长文本哪个区域"，不受人为的伪位置顺序约束。
- **文本侧天然支持变长**：文本 K 侧的 RoPE 无需预设 $N_{\max}$，任意长度的输入都能直接处理，不存在可学习 PE 的长度外推问题。

### 3.4 Process 阶段（Self-Attention：Latent ↔ Latent）

Latent 之间堆叠 $L_{\text{proc}}$ 层 Self-Attention（$M$ 对 $M$），复杂度 $O(M^2)$，$M$ 很小所以代价低廉。

**无论 `use_decoupled_rope` 开关如何，Process 阶段始终使用纯标准多头 Self-Attention，不含任何位置编码。** Latent slot 之间是无序的集合，Process 阶段的作用是让 slot 之间交换信息、消除冗余，不需要位置信息参与。

每层均含 SwiGLU FFN + RMSNorm（Pre-Norm）。

**参数初始化**：Read 和 Process 阶段中所有投影矩阵（$W^Q, W^K, W^V$，以及解耦模式下的 $W^{QC}, W^{KR}$）、可学习 Slot Position Embedding $P^{\text{slot}}$ 均采用**截断高斯分布**初始化，scale = 0.02，截断范围 $[-2\sigma, 2\sigma]$，与 $Z_{\text{base}}$ 保持一致。

### 3.5 Decode 阶段（LLM 生成）

将 Process 阶段输出的 $M$ 个 Memory Token $\tilde{H}$ 视为 Soft Prompt，与提示词拼接后送入冻结的 LLM Decoder：

$$\text{Input} = [\texttt{<MEM>}, \tilde{H}_1, \dots, \tilde{H}_M, \texttt{</MEM>}, E_P]$$

其中 `<MEM>` 和 `</MEM>` 是新增的特殊 Token，用于标示 Memory 区域的边界。LLM Decoder 全程冻结，仅做前向推理生成。

---

## 4. 训练方案

### 4.1 可训练 vs 冻结参数

| 模块 | 状态 |
|:---|:---|
| Qwen3 Embedding Table | **冻结** |
| $Z_{\text{base}}$（可学习 Latent） | **可训练** |
| Perceiver IO 全部参数（Cross-Attn, Self-Attn, FFN, $P^{\text{slot}}$） | **可训练** |
| QueryMapper + $\alpha$（Prompt Bias 通路） | 阶段一**冻结**（输出恒为零），阶段二**可训练** |
| LLM Decoder | **冻结** |

### 4.2 阶段一：文本补全预训练

**目的**：让 Memory Token 学会无损保留长文本的通用语义。

- **开关状态**：`use_prompt_bias=False`（无 prompt），`use_decoupled_rope` 可选（建议与最终推理一致）。
- **数据**：大规模无监督长文本语料，每条切分为前缀 $X_{\text{prefix}}$（送入压缩）和续写 $X_{\text{cont}}$（作为生成目标）。
- **流程**：$X_{\text{prefix}} \to \text{Embed} \to \text{Perceiver IO}(Z_{\text{base}}) \to \tilde{H} \to \text{LLM Decoder} \to \hat{X}_{\text{cont}}$
- **Loss**：仅对 $X_{\text{cont}}$ 部分计算 Next-Token Prediction 交叉熵。

### 4.3 阶段二：QA 指令微调

**目的**：激活 Prompt-Guided Bias，学会定向压缩。

- **开关状态**：`use_prompt_bias=True`，`use_decoupled_rope` 同阶段一。
- **数据**：`(Context, Question, Answer)` 三元组。
- **流程**：
  1. $E_P = \text{Embed}(\text{Question})$，QueryMapper 生成偏置 $B$。
  2. $Z = Z_{\text{base}} + \alpha \cdot B$（训练开始时 $\alpha = 0$，平滑过渡）。
  3. Perceiver IO 以带问题偏置的 $Z$ 对 Context 定向压缩，输出 $\tilde{H}_{QA}$。
  4. $[\tilde{H}_{QA}, E_P] \to \text{LLM Decoder} \to \hat{A}$。
- **Loss**：仅对 Answer 部分计算交叉熵。

---

## 5. 关键超参数建议

| 超参数 | 含义 | 建议范围 |
|:---|:---|:---|
| $M$ | Memory Token 数量 | 64 ~ 256 |
| $D$ | 隐藏维度（与 Qwen3 Embed 对齐） | 2048 / 3584 |
| $n_h$ | 注意力头数 | 16 ~ 32 |
| $d_h$ | 每头维度 | $D / n_h$ |
| $d_R$ | 位置通道维度（仅解耦模式） | 64 ~ 128 |
| $P^{\text{slot}}$ | 可学习 Slot Position Embedding（仅解耦模式，**可训练**） | $M \times d_R$ |
| $N_{\max}$ | 可学习 PE 最大长度（仅 baseline 模式） | 32768 ~ 131072 |
| $L_{\text{proc}}$ | Process 层数 | 4 ~ 8 |
| $D_{\text{mid}}$ | QueryMapper 中间维度 | $D / 2$ |

---

## 6. 方案优势总结

1. **极低编码开销**：长文本仅过 Embedding Lookup（$O(N)$），不经过任何 Transformer Self-Attention 层。
2. **解耦位置感知**：相比原始 Perceiver 的可学习 PE（一次注入、逐层稀释、需预设最大长度），解耦模式在 Read 阶段的每层 Cross-Attention 中持续注入位置信息——文本侧通过 RoPE 保留真实绝对位置，Latent 侧通过可学习 Slot Position Embedding 获得自由的区域偏好能力，内容与位置通道彼此不干扰。Process 阶段不引入位置信息，尊重 Latent slot 的无序集合语义。
3. **零初始化平滑过渡**：Prompt Bias 通路从恒等出发，阶段一的预训练成果零损失地继承到阶段二。
4. **完全模块化**：两个开关产生四种配置，baseline 与 full model 共享同一代码路径，便于消融实验。
