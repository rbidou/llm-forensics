# LLM Forensics

Weight-level forensics for causal language models. Given two models, LLM Forensics
answers:

| Question | Analysis |
|---|---|
| How much, and where, did a fine-tune move the weights? | **Weight Diff** |
| Which tokens do the heaviest updates promote, suppress or react to? | **Token Shifts** |
| How does a Hub model compare with its declared parent and children? | **Hub Lineage** |
| Is this model derived from another one, even undeclared or disguised? | **Provenance** |
| Where do the two models' predictions differ, and what does the fine-tune write? | **Behaviour Probe** |

It reads full models, merged fine-tunes, LoRA adapters, training checkpoints,
**GGUF** files and **bitsandbytes** 4/8-bit checkpoints, locally or straight from the
Hugging Face Hub (by HTTP Range requests: only the tensors needed are fetched).
Every analysis produces a self-contained HTML report and a JSON report.

The motivation is security: auditing where a model comes from, detecting undeclared
derivatives, and understanding what an adversarial fine-tune taught a model. Every
similarity is reported against a null baseline, so a number can be read as evidence.
It has been validated on real models, including an adversarial fine-tune whose
injected payload the Behaviour Probe exposes.

**Manual**: [llm-forensics-manual.pdf](llm-forensics-manual.pdf) (quick start,
user guide, concepts, implementation and parameter reference). The web interface
carries its GUI usage, concepts and parameter chapters on its **Documentation** page,
with the PDF for download.

---

## Installation

Python 3.11+ (developed on 3.12, Windows 11; the code is OS-agnostic).

```
python -m venv venv
venv\Scripts\activate                      # Linux/macOS: source venv/bin/activate

# torch first, from the wheel index matching your hardware
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU
#   or a CUDA build: https://pytorch.org/get-started/locally/

pip install -r requirements.txt
python run.py check                        # what this machine can do
```

`run.py check` lists the platform, the GPU, the installed packages and which
features are available; a missing package disables only the features that need it.

Private or gated Hub models: set `HF_TOKEN`, or log in once with `huggingface-cli login`.

## Usage

```
python run.py                       # interactive terminal interface (default)
python run.py web                   # browser interface on http://127.0.0.1:5000 (address set in Setup)
python run.py web --port 9000       # another port for this run only
```

Both interfaces offer the same menus: **Weight Diff, Token Shifts, Hub Lineage,
Provenance, Behaviour Probe, Reports, Environment, Setup**. Setup sets the address and port
the web server listens on (stored in `settings.json`, applied at the next start). In the terminal, `back`, `home`
and `exit` work at every prompt and Ctrl+C cancels a running analysis.

### Model references

Wherever a model is asked for, give a Hub id or a local path:

| Reference | Read as |
|---|---|
| `Qwen/Qwen2.5-Coder-1.5B` | Hub model, read tensor by tensor (no full download) |
| `D:\models\my-merged-model` | local safetensors folder (single file or sharded) |
| a folder with `adapter_config.json` | LoRA adapter, applied to its base on the fly (the base can be overridden) |
| `output\checkpoint-500` | training checkpoint (adapter or full); a run folder offers its checkpoints |
| a `.gguf` file, or a folder / Hub repo of GGUF files | llama.cpp model, dequantized; with several files you pick one (most precise first) |
| a folder whose config says `bitsandbytes` | NF4 / FP4 / int8 checkpoint, dequantized without bitsandbytes |

Refused on purpose: pickle weights (`.bin`, `.pt`) are **never loaded**, because
unpickling can execute code and the models under investigation are untrusted; GPTQ,
AWQ and other quantization methods, DoRA and non-LoRA PEFT methods are named and
refused.

### Reading the results

**Weight Diff** — per tensor: relative change ‖ΔW‖/‖W‖ (typical LoRA 1e-3 – 5e-2;
above 0.1 is heavy), cosine, max |Δ|, fraction of elements changed, and for LoRA the
exact rank of the update. When a model is quantized, each tensor also gets its
**noise floor** — the change quantization alone would produce — and a *beyond noise*
flag (more than twice the floor). Floors are exact for bitsandbytes and llama.cpp
legacy types, estimated (and labelled) for K/IQ-quants.

**Token Shifts** — the heaviest updates by largest singular value σ₁. An update that
*writes* to the residual stream (`o_proj`, `down_proj`) is read through the
unembedding: the tokens it promotes (+) and suppresses (−). One that *reads* from it
(`q/k/v/gate/up`) is matched against the embeddings: the tokens that trigger it
(meaningful mostly in early layers). Each list comes with a z-score against random
directions: |z| < 3 is noise.

**Hub Lineage** — declared parents (`base_model:*` tags) and children (fine-tunes,
adapters, quantizations ranked by downloads), then a Token Shifts comparison of each
chosen pair.

**Provenance** — candidate parents (explicit, declared, or searched on the Hub) are
screened cheaply, then the best are compared in depth on the first layers and the
embeddings, every metric against a shuffled null. Verdicts:

| Verdict | Meaning |
|---|---|
| IDENTICAL | same weights (re-upload, rename) |
| DERIVED, same basis | fine-tune or continued training (raw cosine ≥ 0.8) |
| RELATED | heavy continued training, merge or partial reuse (raw cosine ≥ 0.3) |
| DERIVED, transformed basis | weights rotated or permuted to hide the origin; caught by invariant "token circuits" |
| NO WEIGHT-LEVEL EVIDENCE | independent, or distilled (distillation leaves no weight trace) |

**Behaviour Probe** — runs prompts through both models (GPU when they fit, split
GPU/CPU otherwise): KL divergence per position, the most divergent positions with
their context, next-token predictions, tokens favoured or avoided, hidden-state drift
per layer, and greedy generations side by side. The target prompts' KL is compared
with 12 neutral control prompts (z-score): high z means the fine-tune changed
behaviour on *those* prompts specifically. For broad fine-tunes that change
everything, the generations and the hottest positions are the evidence. Prompt files:
`.txt` (one per line), `.json` / `.jsonl` (strings, objects with a field, or
chat-format `{"messages": [...]}` datasets). An interactive session keeps both models
loaded for ad-hoc prompts.

### Reports

Every analysis is saved automatically under `reports/<id>/` as `report.html`
(self-contained, no scripts, prints in a light theme) and `report.json`. The Reports
menu lists, opens, re-renders and deletes them; the web interface downloads them.

### Security notes

- The web interface binds to 127.0.0.1 by default, refuses foreign `Host` headers (DNS
  rebinding: only loopback names, the bound address and, bound to `0.0.0.0`, this
  machine's own addresses are accepted) and cross-origin state-changing requests, and
  sends a strict CSP. Do not expose it on a network: it reads any model path you type,
  and has no authentication.
- Token texts and model metadata are treated as hostile: everything rendered is
  escaped; HTML reports carry no script.

---

## Methods

### Shared concepts

**ΔW of a LoRA adapter.** PEFT stores, for each adapted linear layer of weight `W`
(out × in), `lora_A` (r × in) and `lora_B` (out × r). The update is
`ΔW = (B @ A) · scale`, with `scale = alpha / r`, or `alpha / sqrt(r)` under rsLoRA.
Embedding layers use `lora_embedding_A` / `lora_embedding_B` and their update is
transposed: `ΔW = (B @ A)ᵀ · scale`. The adapter key
`base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight` adapts the base tensor
`model.layers.0.self_attn.q_proj.weight` (strip `base_model.model.` once, drop
`lora_A`). Any other tensor in the adapter is a module saved in full
(`modules_to_save`, typically `lm_head` or `embed_tokens`): it replaces the base
tensor. An adapter is applied to its base on the fly, tensor by tensor, so it can be
analysed wherever a full model is expected. Its updates are rebuilt exactly from the
factors, and their SVD and rank are computed exactly through two QR decompositions of
the thin factors. DoRA (`lora_magnitude_vector`) is refused: ΔW alone would be wrong.

**Relative change** `‖ΔW‖ / ‖W‖` (Frobenius norm). A typical LoRA gives 1e-3 to 5e-2;
above 0.1 is a heavy shift. A tensor counts as changed when ‖ΔW‖ ≥ 1e-6. A tensor
whose base is all zeros has no relative change (shown as —).

**Writers and readers** (the residual-stream view of a transformer). In
`torch.nn.Linear` the weight is (out, in); with `ΔW = U S Vᵀ`, `U[:, 0]` lives in the
output space and `V[:, 0]` in the input space.

- *Writers* add to the residual stream: `o_proj`, `down_proj` (also `out_proj`,
  `c_proj`, `w2`, `wo`, `dense_4h_to_h`). Their output direction `u`, pushed through
  the final norm and the unembedding, gives the tokens the update **promotes** (+) and
  **suppresses** (−). This is the logit lens.
- *Readers* read from it: `q/k/v_proj`, `gate/up_proj` (also `qkv_proj`,
  `gate_up_proj`, `query_key_value`, `c_attn`, `c_fc`, `w1`, `w3`). Their input
  direction `v`, compared by cosine with the token embeddings, gives the tokens that
  **trigger** the update. This is only really meaningful in early layers, where the
  residual stream is still close to the embeddings.
- **The sign of an SVD is arbitrary.** The "+" and "−" lists are the two ends of one
  axis, not "good" and "bad". Signs are normalised (largest |u| component positive)
  so that runs are reproducible.

**Unembedding through the final norm.** The unembedding matrix is `lm_head.weight`
when the model has one, else the token embedding (tied models), multiplied
element-wise by the final norm gain. Readers are always compared with the real
*input* embedding (`embed_tokens`, `wte` or `word_embeddings`), so untied GPT-2-style
models are handled correctly.

**Null baselines.** Every similarity is reported against what chance gives with the
same data: rows, columns or tokens shuffled, random directions, or neutral prompts.
All randomness is seeded, so results are reproducible. Read a number against its
null (or its z-score), never on its own.

### Weight Diff

Every tensor present in both models is compared. When the tuned model is an adapter
on that same base, only the adapted tensors and the modules saved in full are
visited. Per tensor the analysis reports the relative change, `cos(W, W + ΔW)`,
max |Δ|, the fraction of elements that moved (|Δ| > 1e-6) and, for LoRA, the exact
rank of the update (at most `r`). It then gives the mean relative change per module
type and per layer. A tensor found only in the tuned model, or with a different shape,
is counted as unmatched and noted.

Interpretation:

- A full fine-tune, or a model merged correctly, shows exact zeros on every tensor it
  did not train; an adapter compared with its base shows only the targeted modules.
- A model merged into a **4-bit** base and then saved differs slightly on *every*
  tensor (quantization noise). Merge into the bf16 base, or read such a comparison
  against the noise floor (see Quantized models).

### Token Shifts

Every update is reduced to its leading singular triplet (σ₁, u, v). LoRA updates use
an exact SVD through their factors; full updates use a randomized SVD (rank 6, 4 power
iterations, seeded per tensor so the result does not depend on visiting order). The
`n` heaviest 2-D updates by σ₁ are kept, together with the **share** σ₁² / ‖ΔW‖²: LoRA
updates are usually dominated by one or two directions. Each kept update is then
projected onto the vocabulary, as a writer or a reader (see above).

**Null baseline:** for each end of the axis, the mean of its top-k scores is compared
with the same statistic for 20 random unit directions, which gives a z-score. |z| < 3
is noise level.

For every embedding or `lm_head` matrix that changed, the analysis also lists the
tokens whose row moved most (relative change per row). This is direct token-level
evidence for full fine-tunes. Finally, a per-layer profile of the mean relative change
shows where the training concentrated.

Interpretation rule: a token that shows up in a writer projection (static) **and** among
the tokens promoted by the Behaviour Probe (dynamic) is a solid finding.

### Behaviour Probe

The same input goes through the base and the tuned model; per position:

`KL(tuned ‖ base) = Σ p_tuned · (log p_tuned − log p_base)` over next-token distributions

computed on the shared vocabulary. With a LoRA adapter only one model is in memory: the
base outputs come from the same model with the adapter disabled. Over all positions, the
mean log-probability shift of each vocabulary token gives the tokens **promoted** and
**suppressed**. Hidden-state drift per layer is the mean `1 − cos(h_base, h_tuned)`;
entry 0 is the embeddings. The chat template is applied as a user turn when the
tokenizer has one, and inputs are truncated to 512 tokens. Generations are greedy.

**Null baseline:** 12 neutral control prompts (everyday questions) go through the same
comparison. The target prompts' mean KL is reported as a z-score against the controls'
per-prompt KLs. A targeted fine-tune (a backdoor, a topic) shifts the target prompts
more than neutral ones. A broad fine-tune (instruction tuning, many epochs) shifts
everything, and z stays near 0: then the generations and the most divergent positions
carry the evidence.

**Loading and placement:** full models load with safetensors only; GGUF models load
through transformers' GGUF support; bitsandbytes checkpoints are dequantized by the
toolkit and loaded into a plain model, so bitsandbytes is not needed. On a GPU the
models run in bf16, otherwise in float32 on the CPU. Both models go on the GPU if their
weights fit in 85 % of its free memory, otherwise the tuned model goes on the GPU and
the base on the CPU, otherwise both run on the CPU. If the two tokenizers differ,
token IDs are not comparable: KL and token statistics are disabled and only the
generations run.

### Hub Lineage

Discovery uses Hub metadata only; nothing is verified on the weights at this stage.

- **Parents** come from the `base_model:<relation>:<repo>` tags (relations `finetune`,
  `adapter`, `merge`, `quantized`), falling back on the model card's `base_model`.
  `finetune`, `adapter` and `quantized` parents are compared. The relation is
  self-declared: SmolLM2-135M-Instruct, a full fine-tune, declares its base as
  `quantized`. A merge is compared with its first listed parent only.
- **Children** come from `list_models(filter="base_model:{adapter|finetune|quantized}:<id>")`,
  ranked by downloads and limited to `max_variants`.
- **What a repository is** comes from its files and config: an adapter
  (`adapter_config.json`), GGUF files (the most precise file is used), bitsandbytes,
  another quantization (skipped: GPTQ, AWQ…), or full safetensors. Full and GGUF
  children larger than `max_size_gb` are skipped. Every skip is listed with its reason.

Each chosen pair is then compared with Token Shifts, reading both models by HTTP Range.
A failing pair is recorded and the others continue.

### Provenance

**Principle.** Independently trained models with the same architecture have
near-orthogonal corresponding weights (cosine ≈ 0, random initialisation in very high
dimension). A fine-tune stays above 0.99. Early layers and embeddings move least
during fine-tuning, so they are the best fingerprint. Some transformations preserve
the model's function but break raw cosine:

- permuting MLP neurons or attention heads: countered by permutation-invariant metrics
  (best-match row cosine, sums over heads);
- an orthogonal rotation `R` of the residual stream applied consistently (SliceGPT /
  QuaRot style: `E → E R`, readers `W → W R`, writers `W → Rᵀ W`, norm gains folded
  into the weights): countered by quantities where `R` cancels, such as `E Eᵀ` or
  `E W_qᵀ W_k Eᵀ`;
- distillation shares no weights at all. Weight-level tests must answer "no evidence";
  only behavioural fingerprinting can detect it.

**Pipeline.**

1. **Suspect**: any supported model. A LoRA adapter is analysed as its base plus the
   adapter (the base can be overridden). A quantized model is dequantized first:
   quantization lowers raw cosines only slightly, so the thresholds still apply.
2. **Candidates**, each tagged with its origin:
   - *explicit*: given by you;
   - *declared*: `base_model:*` tags walked upwards (depth 4), plus the base an adapter
     declares;
   - *search*: Hub models of the same `model_type` (or a free-text search) ranked by
     downloads, GGUF repositories excluded, and only root models (no `base_model` tag)
     unless derived ones are included. Instruct models carry a `base_model` tag, so they
     are excluded by default.

   Candidates whose hidden size, MLP size, head counts or head dimension differ are
   skipped. The number of layers and the vocabulary size may differ (pruned or
   vocabulary-extended derivatives).
3. **Stage 1 — screen** (in parallel, a few MB per candidate): the raw cosine of the
   first 64 rows of the layer-0 `q`, `k`, `v`, `gate` and `up` matrices (their mean),
   plus the Pearson r of the `input_layernorm` gains.
4. **Stage 2 — deep comparison** of the best `top_k` screened candidates (default 3),
   plus every explicit or declared one, on layers `0 … layers−1`:
   - per tensor:
     - `raw_cos`, with a null where the candidate's rows are shuffled (mean of 3 draws);
     - `rel_d = ‖W_s − W_c‖ / ‖W_c‖`;
     - for `gate` and `up`, `nn_row`: the mean, over 512 sampled suspect neurons, of the
       best cosine against any candidate neuron. It is invariant to neuron permutation;
       its null shuffles the candidate's columns (3 draws);
     - `spectrum`: the singular-value similarity `1 − ‖s_a − s_b‖ / ‖s_a‖`, with norm
       gains folded in. It is invariant to rotation and permutation but weak, because
       same-architecture models have similar spectra;
   - the embeddings slice (rows 1000 to 2023 by default): raw cosine, with a row-shuffled null;
   - **token circuits**, on layer 0 and the embedding slice, using RMS-normalised
     embeddings `En` with the norm gains folded into the inputs:
     - `emb_gram = En Enᵀ`
     - `qk = Σ_heads q_h k_hᵀ` (RoPE is the identity at position 0; KV heads are
       expanded for grouped-query attention)
     - `ov = (v W_oᵀ) Enᵀ`
     - `mlp = (act(X W_gateᵀ) ⊙ X W_upᵀ) W_downᵀ Enᵀ`

     All four are token × token matrices in which a residual rotation cancels and
     neuron or head permutations are summed out. Each is compared by the Pearson r of
     its off-diagonal entries, against a null where the candidate's tokens are permuted
     consistently (10 draws), which gives the mean, sd and z. Biases are ignored: this
     is a fingerprint, not an exact forward pass. At least 64 tokens are needed.
5. **Verdict** (heuristic thresholds, checked in this order):

| Rank | Condition | Verdict |
|---|---|---|
| 5 | median raw_cos > 0.99999 and max rel_d < 1e-5 | IDENTICAL weights |
| 4 | median raw_cos ≥ 0.8 | DERIVED, same basis (fine-tune / continued training) |
| 3 | median raw_cos ≥ 0.3 | RELATED (heavy continued training, merge or partial reuse) |
| 2 | token circuits mean r ≥ 0.8 and min z ≥ 10 | DERIVED, transformed basis (rotation / permutation) |
| 2 | nn_row(up, layer 0) ≥ 0.8 and z ≥ 10 | DERIVED, permuted MLP neurons |
| 0 | otherwise | NO WEIGHT-LEVEL EVIDENCE (independent, or distilled) |

6. **Summary**: candidates are sorted by rank, then raw cosine, then circuits. The best
   match (rank ≥ 2) is named, with warnings when it differs from the declared base, or
   when a Hub suspect declares no lineage at all.

**Caveat on invariants.** `emb_gram`, and to a lesser extent the other circuits, can
correlate moderately between *unrelated* models, because token semantics are shared
across models. Always read r against its null and, ideally, against an unrelated
candidate of the same architecture. `emb_gram` is excluded from the verdict.

These thresholds were checked on real models without recalibration:

| Comparison | Verdict |
|---|---|
| a LoRA fine-tune vs its base | DERIVED, same basis (raw 0.9985) |
| Qwen2.5-Coder vs general Qwen2.5 (Coder is continued pretraining of Qwen2.5) | RELATED (raw 0.36) |
| an independent model of the same architecture | NO EVIDENCE (raw 0.0002) |
| a rotated copy | DERIVED, transformed basis (circuits r 1.000) |

### Quantized models

**GGUF.** Tensors are dequantized to float32 one at a time. Every type the `gguf`
package can decode is supported: the legacy, K, IQ and ternary types and MXFP4. Rows
are cut from the raw blocks before dequantizing, so partial reads stay cheap.

- llama.cpp tensor names are mapped back to Hugging Face names through the `gguf`
  package's name table, keeping every alias, so a base of any family matches.
- `convert_hf_to_gguf.py` permutes the rows of `q_proj` / `k_proj` for llama-family
  architectures (rotary layout); that permutation is reversed.
- A config is rebuilt from the GGUF metadata, and the tokenizer can be read from the
  file itself.
- A folder or Hub repository with several GGUF files asks which one to read; unattended
  analyses take the most precise.

**bitsandbytes.** 4-bit NF4 / FP4 weights (packed codes, per-block scales, optional
double quantization of the scales) and 8-bit int8 weights (row scales `SCB`) are
dequantized in pure torch. The result is identical to bitsandbytes' own, which rounds
it to the checkpoint's dtype.

**Noise floor.** Comparing a full-precision model with a quantized one mixes two
effects: what fine-tuning changed, and what quantization destroyed. The second is
measured per tensor by quantizing the *base* tensor with the same scheme and
dequantizing it again; the relative error obtained is the change the tensor would show
had fine-tuning not touched it. A change is **beyond noise** when it exceeds twice that
floor and is above rounding level (relative change > 1e-5).

- Floors are exact for bitsandbytes and for the llama.cpp types the `gguf` package can
  quantize (legacy, ternary, F16/BF16).
- K- and IQ-quants have no Python quantizer, so their floor is *estimated*: a uniform
  quantizer of the same bit width on blocks of 32, and labelled as such. On real
  q4_K_M and q5_K files the estimate is within −9 % / +3 % of the measured error.
- When both models are quantized, their floors combine in quadrature. An adapter
  compared with its own base has no floor, because its update is exact.

---

## Limits

- Tensor naming is Llama-style (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`;
  fused `qkv_proj` and `gate_up_proj` are split). Provenance does not map GPT-2 or
  Falcon-style blocks.
- Token circuits assume the suspect and the candidate share token IDs; a warning is
  given when the vocabulary sizes differ.
- Hub Range reads are one HTTP request per tensor: fine for provenance (a few tensors),
  latency-bound for whole models (about 2.5 MB/s measured). They also depend on the
  Hub storage honouring `Range` (LFS/CDN and Xet do).
- Hub Lineage relies on Hub metadata being declared and correct; use Provenance when it
  is not.
- GGUF files on the Hub are downloaded whole (into the HF cache); safetensors are read
  by Range.
- The Behaviour Probe's control z-score is uninformative for broad fine-tunes (see above).
- Hub lineage discovery and model loading for the Behaviour Probe cannot be cancelled
  midway; a cancel takes effect when they finish.
- Not readable: pickle weights (by design), GPTQ, AWQ, EXL2 and other quantization
  methods, DoRA and non-LoRA PEFT methods.

---

Author: Renaud Bidou (rbidou@gmail.com)

Licensed under the MIT License — see [LICENSE](../LICENSE).
