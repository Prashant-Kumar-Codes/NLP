*Que*: **RNNs are supposed to work with dynamic input sizes, which is why we don't use ANNs. So why do we use padding and truncation to make inputs the same size?**

Mathematically, an RNN/LSTM does **not** care about sequence length. Because it processes text step-by-step recursively ($h_t = f(h_{t-1}, x_t)$), a single RNN can process a sentence of 3 words, then a sentence of 20 words, then a sentence of 7 words—**without any padding at all**.

So why do we use padding and truncating in practice?

---

### The Real Reason: **GPU Parallelism & Batching**

Computers (especially GPUs) do not process sentences one-by-one (`batch_size = 1`). Processing one sentence at a time is **extremely slow** because GPU CUDA cores sit idle waiting for memory transfers.

To make training 100x faster, we process text in **batches** (e.g., `batch_size = 64` sentences simultaneously).

#### 1. PyTorch Tensors MUST be Rectangular Matrices
In PyTorch/NumPy, a batch is stored as a single 3D Tensor matrix:
$$\text{Shape: } [\text{batch\_size}, \text{sequence\_length}, \text{embedding\_dim}]$$

A tensor matrix **cannot have jagged rows** (where Row 1 has length 3, Row 2 has length 15, and Row 3 has length 8). Every row in a matrix must have the **exact same dimensions** so the GPU can perform fast matrix multiplications.

```text
❌ Cannot put into a single GPU Tensor (Jagged):
Batch Row 1: ["Great", "movie"]                    (Length 2)
Batch Row 2: ["I", "hated", "this", "film", "bad"] (Length 5)

✅ Padded to uniform shape [2, 5] for GPU Tensor:
Batch Row 1: ["Great", "movie", "<pad>", "<pad>", "<pad>"] (Length 5)
Batch Row 2: ["I", "hated", "this", "film", "bad"]         (Length 5)
```

---

### Summary: Theory vs. Practice

| | Theoretical RNN (Math) | Practical Deep Learning (Hardware) |
| :--- | :--- | :--- |
| **Execution** | Sample by sample (`batch_size = 1`) | Batch of samples (`batch_size = 64`) |
| **Sequence Length** | Truly dynamic (No padding needed) | Uniform tensor matrix (Padding & Truncating required) |
| **Speed** | 🐌 Very Slow (GPUs underutilized) | ⚡ Fast (Fully parallel GPU matrix multiplication) |

---

### 💡 Advanced Tip: Dynamic Padding
Instead of padding all dataset samples to a global fixed length like `40`, production systems use **Dynamic Padding** (or `torch.nn.utils.rnn.pack_padded_sequence`):
* Batch 1 pads to length 12 (longest sentence in Batch 1).
* Batch 2 pads to length 7 (longest sentence in Batch 2).

This saves memory while keeping GPU batching fast!