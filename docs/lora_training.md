# LoRA Data Preparation: Best Practices

This guide outlines the optimal strategy for preparing training datasets when training a LoRA adapter (specifically for coding libraries and documentation).

## 1. The Golden Rule: Quality > Quantity

When training a LoRA, the goal is **instruction tuning**, not knowledge injection. 
* **Do not** feed raw source code or raw documentation text files.
* **Do** feed "Chat Pairs" (Question & Answer) that teach the model *how* to use the knowledge.

If you feed raw text, the model becomes a document auto-completer. If you feed Q&A pairs, it becomes an intelligent assistant.

---

## 2. Data Format & Templates

Always use the **Chat Template** native to the base model you are training (e.g., ChatML for Qwen, Alpaca for Llama 2, etc.). If you mismatch the template, the model will output garbage or stop tokens incorrectly.

### Example: Qwen / ChatML Format
*Used for Qwen2.5, Yi, and modern generic models.*

```text
<|im_start|>user
How do I initialize the 'TensorArray' struct from a standard vector?<|im_end|>
<|im_start|>assistant
You can use the `TensorArray::new()` method. This method takes ownership of a `Vec<T>` and returns a new tensor instance.

Example:
```rust
let data = vec![1.0, 2.0, 3.0];
let tensor = TensorArray::new(data);

```

<|im_end|>

```

---

## 3. The "Rule of Three" for Code

**Never** repeat the exact same code snippet with the same question multiple times. This causes "overfitting" (rote memorization). instead, use the **Rule of Three** to teach the *concept*:

### A. The Basic Usage ("Hello World")
* **Q:** "How do I create a TensorArray?"
* **A:** Shows the simplest constructor with hardcoded integers.
    ```rust
    let t = TensorArray::new(vec![1, 2, 3]);
    ```

### B. The Type Variation
* **Q:** "Can I use floating point numbers?"
* **A:** Shows the same function, but with `f32` data to teach type flexibility.
    ```rust
    let t = TensorArray::new(vec![1.5, 2.5, 3.5]);
    ```

### C. The Contextual Usage
* **Q:** "How do I convert user input into a Tensor?"
* **A:** Shows the function being used inside another function or logic block.
    ```rust
    fn parse_input(input: Vec<i32>) -> TensorArray {
        if input.is_empty() { panic!("Empty input"); }
        TensorArray::new(input)
    }
    ```

---

## 4. Automating Dataset Creation (The "Lazy" Pro Way)

Manually writing 500+ Q&A pairs is tedious. The best practice is to use a larger, smarter model (like GPT-4, Claude 3.5, or a local Llama-3-70B) to generate the synthetic data for you.

### Workflow:
1.  **Chunk your Docs:** Split your new library documentation (e.g., Rust `.md` files) into small chunks (1 function or struct per chunk).
2.  **Prompt the Big Model:**
    > "I am going to paste a section of documentation for a Rust library. Please generate 3 distinct Q&A pairs based on this text.
    > 1. One conceptual question (What is this?)
    > 2. One coding implementation question (How do I use it?)
    > 3. One edge-case question (What happens if...?)
    > Output the result in ChatML format."
3.  **Review:** quickly skim the output to ensure the code is correct.
4.  **Tokenize:** Run the generated file through `dataset-format`.

### Why this works:
Big models already know the *syntax* of popular languages (Rust, Python, C++). They just need your specific library's *names* and *rules*. They do the heavy lifting of formatting, punctuation, and tone, saving you hours of work.