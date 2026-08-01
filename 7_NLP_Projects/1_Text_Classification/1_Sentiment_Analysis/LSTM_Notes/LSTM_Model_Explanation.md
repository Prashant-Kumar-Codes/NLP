# Bidirectional LSTM (`SentimentBidirectionalLSTM`) Architecture Explanation

This document provides a comprehensive, step-by-step technical breakdown of the **`SentimentBidirectionalLSTM`** PyTorch neural network implemented in `1_Twitter_Sentiment_Analyis.ipynb` for 3-class Twitter sentiment classification (`Negative: 0`, `Neutral: 1`, `Positive: 2`).

---

## 1. High-Level Architecture Overview

The model is a **2-layer Bidirectional Long Short-Term Memory (BiLSTM)** network paired with a Fully Connected (Linear) classification head.

```text
Input Tokens (IDs)       [ Batch Size = 64, Max Len = 40 ]
       │
       ▼
Embedding Layer          [ 64, 40, 120 ]  (vocab_size = 9213, embed_dim = 120, padding_idx = 0)
       │
       ▼
Spatial Dropout          [ 64, 40, 120 ]  (p = 0.3)
       │
       ▼
Stacked 2-Layer BiLSTM   Forward & Backward processing across 40 time steps
       │                 lstm_out : [ 64, 40, 296 ]  (148 hidden_dim * 2 directions)
       │                 hidden   : [ 4, 64, 148 ]   (2 layers * 2 directions)
       ▼
State Concatenation      [ 64, 296 ]      Concat(Layer2 Forward, Layer2 Backward)
       │
       ▼
FC Dropout               [ 64, 296 ]      (p = 0.3)
       │
       ▼
Linear Layer (Classifier)[ 64, 3 ]        Output logits for 3 sentiment classes
```

---

## 2. Project Hyperparameters & Configuration

The model in `1_Twitter_Sentiment_Analyis.ipynb` is instantiated with the following exact parameters:

| Hyperparameter | Variable Name | Project Value | Description |
| :--- | :--- | :--- | :--- |
| **Vocabulary Size** | `len(vocab)` | `9213` | Total number of unique tokens in vocabulary (including `<pad>`:0, `<unk>`:1) |
| **Embedding Dim** | `EMBED_DIM` | `120` | Size of dense word vector representation |
| **Hidden Dim** | `HIDDEN_DIM` | `148` | Number of LSTM units per direction in each layer |
| **Output Dim** | `OUTPUT_DIM` | `3` | Number of target sentiment classes (`Negative: 0`, `Neutral: 1`, `Positive: 2`) |
| **LSTM Layers** | `NUM_LAYERS` | `2` | Number of vertically stacked LSTM layers |
| **Dropout Rate** | `DROPOUT` | `0.3` | Regularization probability for inter-layer and FC dropout |
| **Batch Size** | `BATCH_SIZE` | `64` | Number of sequences processed in parallel per step |
| **Sequence Length** | `MAX_LEN` | `40` | Uniform padded/truncated text sequence length |

---

## 3. Component Breakdown (`__init__`)

```python
class SentimentBidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(SentimentBidirectionalLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
```

### Detailed Layer Descriptions:

1. **`self.embedding` (`nn.Embedding`):**
   * Maps integer token IDs into `120`-dimensional continuous vector space.
   * **`padding_idx=0`:** Tells PyTorch that index `0` corresponds to `<pad>`. Weight gradients for index `0` are initialized to zero and kept at zero throughout training.

2. **`self.lstm` (`nn.LSTM`):**
   * **`input_size=120`:** Accepts `120`-dimensional embedding vectors.
   * **`hidden_size=148`:** Each directional LSTM unit has `148` hidden units (state vectors $h_t$ and $c_t$ have length `148`).
   * **`num_layers=2`:** Two LSTMs are stacked vertically. Layer 1 processes word embeddings; Layer 2 takes Layer 1's sequence outputs as its input.
   * **`batch_first=True`:** Expects inputs in shape `[batch_size, sequence_length, features]`.
   * **`dropout=0.3`:** Applies dropout between Layer 1 and Layer 2 outputs to prevent co-adaptation.
   * **`bidirectional=True`:** Runs two parallel LSTMs per layer—one reading left-to-right (forward) and one reading right-to-left (backward).

3. **`self.fc` (`nn.Linear`):**
   * Maps concatenated bidirectional features (`148 * 2 = 296`) to the final `3` class logits.

4. **`self.dropout` (`nn.Dropout`):**
   * Randomly zeroes elements with probability `p=0.3` during training for regularization.

---

## 4. Forward Pass & Granular Tensor Transformations (`forward`)

Given an input batch `text` of shape `[64, 40]` (containing 64 tweets, padded/truncated to 40 token IDs):

```python
def forward(self, text):
    embedded = self.dropout(self.embedding(text))  # [64, 40, 120]
    lstm_out, (hidden, cell) = self.lstm(embedded)
    
    # Concatenate forward and backward final hidden state
    hidden_last = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # [64, 296]
    logits = self.fc(self.dropout(hidden_last))
    
    return logits
```

### Step-by-Step Tensor Dimensionality Trace:

#### Step 1: Embedding Lookup & Dropout
* **Input `text`:** `[64, 40]` (Integer Tensor)
* **`self.embedding(text)`:** `[64, 40, 120]`
* **`embedded` after Dropout:** `[64, 40, 120]`

#### Step 2: BiLSTM Processing
* **Input to `self.lstm`:** `embedded` `[64, 40, 120]`
* **`lstm_out` Output:** `[64, 40, 296]`
  * `296` comes from `hidden_dim * num_directions` = `148 * 2`.
  * Contains the concatenated forward and backward hidden states for **every single time step** ($t=1 \dots 40$) of the top layer (Layer 2).
* **`hidden` State Tensor:** `[4, 64, 148]`
  * Shape formula: `[num_layers * num_directions, batch_size, hidden_dim]` = `[2 * 2, 64, 148]` = `[4, 64, 148]`.
  * Tensor indexing breakdown:
    - `hidden[0, :, :]`: Layer 1 **Forward** final hidden state
    - `hidden[1, :, :]`: Layer 1 **Backward** final hidden state
    - `hidden[2, :, :]` (`hidden[-2]`): Layer 2 **Forward** final hidden state
    - `hidden[3, :, :]` (`hidden[-1]`): Layer 2 **Backward** final hidden state
* **`cell` State Tensor:** `[4, 64, 148]` (Internal long-term memory state $c_t$).

#### Step 3: Top-Layer Hidden State Extraction & Concatenation
* **`hidden[-2, :, :]`:** Takes top-layer forward state $\rightarrow$ `[64, 148]`
* **`hidden[-1, :, :]`:** Takes top-layer backward state $\rightarrow$ `[64, 148]`
* **`torch.cat((hidden[-2], hidden[-1]), dim=1)`:** Concatenates along feature dimension (`dim=1`).
* **`hidden_last`:** `[64, 296]`

#### Step 4: Fully Connected Projection
* **`self.dropout(hidden_last)`:** `[64, 296]`
* **`self.fc(...)`:** Projects `[64, 296]` $\rightarrow$ `[64, 3]`.
* **Output `logits`:** `[64, 3]` (Raw unnormalized prediction scores for `Negative`, `Neutral`, and `Positive`).

---

## 5. Forward Pass Approaches: Direct Hidden State vs. Masked Global Average Pooling

In `1_Twitter_Sentiment_Analyis.ipynb`, two forward pass strategies were explored:

### Approach A: Direct Final Hidden State Extraction (Current Active Model)
```python
hidden_last = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
logits = self.fc(self.dropout(hidden_last))
```
* **Pros:** Directly captures the final hidden states from the top layer after reading the sequence in both directions.
* **Mechanism:** Uses `hidden[-2]` (Forward state after processing word $T$) and `hidden[-1]` (Backward state after processing word 1).

### Approach B: Masked Global Average Pooling
```python
lstm_out, _ = self.lstm(embedded)  # [64, 40, 296]

# Create binary mask for non-padding tokens (1 for real words, 0 for pad)
mask = (text != 0).unsqueeze(-1).float()  # [64, 40, 1]

# Zero out padding tokens in lstm_out
masked_out = lstm_out * mask  # [64, 40, 296]

# Average over actual word positions
sum_out = masked_out.sum(dim=1)  # [64, 296]
lengths = mask.sum(dim=1).clamp(min=1)  # [64, 1]
pooled = sum_out / lengths  # [64, 296]

logits = self.fc(self.dropout(pooled))  # [64, 3]
```
* **Pros:** Averages `lstm_out` representations across **only actual non-padded tokens**, ignoring `<pad>` tokens (index `0`).
* **Mechanism:** Prevents sequence representations of short tweets (e.g., 3 words + 37 `<pad>` steps) from decaying over trailing padding steps.

---

## 6. Model Output Interpretation

Given `logits` of shape `[64, 3]`:

```python
# Obtain predicted class indices (0, 1, or 2)
preds = torch.argmax(logits, dim=1)  # Shape: [64]

# Map back to sentiment strings using id2label_mapper
# 0 -> 'negative', 1 -> 'neutral', 2 -> 'positive'
```
* For classification loss, `logits` are passed directly into `nn.CrossEntropyLoss(logits, targets)`, which internally applies `LogSoftmax` followed by `NLLLoss`.
