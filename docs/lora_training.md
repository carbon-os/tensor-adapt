# LoRA Data Preparation: Best Practices

## Understanding Initial Loss

The loss value at step 1 before any training has occurred tells you how familiar the
base model already is with your domain. It is not random — the base model has prior
knowledge that raises or lowers the starting point.

| Initial loss | Meaning |
|---|---|
| `~11.9` | theoretical maximum — model has zero relevant knowledge (uniform random over vocab) |
| `8.0–10.0` | domain is almost entirely foreign — niche library, private API, invented syntax |
| `6.5–8.0` | **model knows the language but not this specific package** — e.g. knows Go, never seen `gomarkdown` |
| `4.5–6.5` | model has partial knowledge — popular library with limited training coverage |
| `2.5–4.5` | model already knows this domain well — mainstream library like `net/http`, `pandas`, `numpy` |
| `<2.5` | model was heavily trained on this exact material — fine-tuning may have limited upside |

### What this means in practice

If your initial loss is in the `6.5–8.0` range (like the gomarkdown run at `7.57`), that
is the ideal LoRA target. The base model provides a strong foundation — it understands
Go syntax, types, interfaces, and conventions — but the specific package names, function
signatures, and extension flags are genuinely new information. The LoRA is doing exactly
what it is designed for: injecting specific knowledge on top of general capability.

If your initial loss is already below `4.0`, the base model likely has reasonable coverage
of your domain. A LoRA will still improve precision and reduce hallucination, but the
gains will be smaller and you need fewer tokens to reach the sweet spot.

If your initial loss is above `9.0`, the domain is very foreign. Consider expanding your
dataset significantly before training — a small Q&A set on a completely unknown domain
will produce a narrow, brittle adapter.

### Common initial loss ranges by domain

| Domain | Typical initial loss | Why |
|---|---|---|
| Popular Python libs (`numpy`, `pandas`, `requests`) | `2.0–4.0` | Saturated in pretraining data |
| Popular Go stdlib (`net/http`, `fmt`, `io`) | `2.5–4.5` | Well represented in open source |
| Mainstream JS frameworks (`React`, `Express`) | `2.0–3.5` | Enormous web presence |
| Niche Go packages (`gomarkdown`, `chi`, `bubbletea`) | `6.5–8.5` | Limited pretraining coverage |
| Private/internal APIs | `8.0–11.0` | Never seen before |
| Invented DSLs or custom syntax | `9.0–11.9` | Completely foreign |

---

## Loss Targets During Training

| Loss | Meaning |
|---|---|
| `>5.0` | undertrained, model barely learned anything |
| `3.5–5.0` | light knowledge injection, will answer but imprecisely |
| `2.0–3.5` | **sweet spot** — learned the domain, still generalizes |
| `1.0–2.0` | strong fit, verging on memorization |
| `<1.0` | memorized, will regurgitate training text verbatim |

Loss is a proxy. A model at 3.8 that correctly answers questions it hasn't seen before
is better than one at 2.0 that only parrots training sentences back.

---

## Token Budget by Dataset Size

The dominant mistake is setting `--tokens` relative to a wall-clock budget rather than
the actual dataset size. The model needs enough passes to converge, but too many passes
on a small dataset causes memorization.

Rule of thumb: **target 150–200 steps**, where one step = `batch × seq` tokens.
With the default `batch=4 seq=1024` that is **4096 tokens per step**.

| Dataset size | Target steps | `--tokens` |
|---|---|---|
| < 20K tokens (tiny, < 200 Q&A pairs) | 150–200 | `800K` |
| 20K–100K tokens (small) | 200–300 | `1.2M` |
| 100K–500K tokens (medium) | 300–500 | `2M` |
| 500K–2M tokens (large) | 500–800 | `4M` |
| > 2M tokens (full dataset) | 800–1500 | `8M+` |

### Examples

```bash
# Tiny dataset — ~12K tokens, 160 Q&A pairs (e.g. gomarkdown)
dataset-format \
  --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
  --input     ./gomarkdown.jsonl \
  --output    ./data/gomarkdown \
  --chatml --no-eos

tensor-adapt \
  --arch   qwen2 \
  --base   ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
  --data   ./data/gomarkdown \
  --domain my/gomarkdown \
  --tokens 800000 \
  --output ./adapters/gomarkdown \
  --device cuda:0

# Small dataset — ~60K tokens, ~800 Q&A pairs
tensor-adapt \
  --tokens 1200000 ...

# Medium dataset — ~300K tokens, ~3000 Q&A pairs
tensor-adapt \
  --tokens 2000000 ...

# Large dataset — ~1M tokens, ~10K Q&A pairs
tensor-adapt \
  --tokens 4000000 ...
```

If loss hasn't reached the sweet spot (`<3.5`) by the end of the run, double `--tokens`
and retrain. If loss went below `1.5`, halve `--tokens` next time.

---

## 1. Quality Over Quantity

The goal of a LoRA is **instruction tuning**, not document ingestion.

- **Do not** feed raw source code, raw docs, or plain text files
- **Do** feed Q&A pairs that teach the model *how to use* the knowledge

Raw text produces a document auto-completer. Q&A pairs produce an intelligent assistant.

---

## 2. Data Format

Always use the chat template native to the base model. Mismatching the template causes
incorrect stop token behaviour and garbage output.

### Qwen / ChatML (Qwen2.5, Yi, and most modern models)

Raw JSONL schema:
```jsonl
{"messages": [{"role": "user", "content": "What is X?"}, {"role": "assistant", "content": "X is ..."}]}
```

After running through `dataset-format --chatml`, each record becomes:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is X?<|im_end|>
<|im_start|>assistant
X is ...<|im_end|>
```

`<|im_start|>` and `<|im_end|>` are special tokens in Qwen2's vocabulary and are emitted
as single token IDs — always use `--chatml --no-eos` for instruct datasets.

---

## 3. The Rule of Three for Code

Never repeat the same snippet with the same question. Instead teach each concept
three ways to prevent rote memorization:

**A. Basic usage** — simplest possible example
```
Q: How do I create a TensorArray?
A: let t = TensorArray::new(vec![1, 2, 3]);
```

**B. Type variation** — same function, different types
```
Q: Can I use floating point numbers with TensorArray?
A: let t = TensorArray::new(vec![1.5f32, 2.5, 3.5]);
```

**C. Contextual usage** — function used inside real logic
```
Q: How do I convert user input into a TensorArray safely?
A: fn parse(input: Vec<i32>) -> TensorArray {
       if input.is_empty() { panic!("empty"); }
       TensorArray::new(input)
   }
```

---

## 4. Generating Datasets with a Larger Model

Manually writing hundreds of Q&A pairs is slow. Use a larger model (GPT-4, Claude,
or a local 70B) to generate synthetic data from your documentation.

### Workflow

1. **Chunk your docs** — one function or struct per chunk
2. **Prompt the large model:**

   > Paste a section of documentation. Generate 3 Q&A pairs:
   > 1. Conceptual — what is this?
   > 2. Implementation — how do I use it?
   > 3. Edge case — what happens if...?
   >
   > Output in JSONL with schema:
   > `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

3. **Review** — skim for code correctness, remove duplicates
4. **Tokenize and train:**

```bash
dataset-format \
  --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
  --input     ./my_dataset.jsonl \
  --output    ./data/my_dataset \
  --chatml --no-eos

tensor-adapt \
  --arch   qwen2 \
  --base   ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
  --data   ./data/my_dataset \
  --domain my/dataset \
  --tokens 800000 \
  --output ./adapters/my_dataset \
  --device cuda:0
```

Large models already know the syntax of common languages — they just need your
library's specific names and rules. They handle formatting and tone, saving hours of work.