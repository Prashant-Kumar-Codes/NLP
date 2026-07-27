# 2.5 Noise Handling in NLP

**Noise Handling** in Natural Language Processing refers to identifying and removing irrelevant, repetitive, uninformative, or corrupted text data from a corpus before downstream modeling.

Unfiltered text often contains dataset-level noise (duplicate rows), word-level noise (misspellings/rare words, repeated character elongations), or frequency noise (over-frequent domain words). Cleaning this noise improves model accuracy, reduces training time, and shrinks memory footprint.

---

## Folder Structure Overview

```
5_Noise_Handling/
├── 1_Rare_Word_Removal/
│   └── 1_Rare_Word_Removal.ipynb
│
├── 2_Frequent_Word_Removal/
│   └── 1_Frequent_Word_Removal.ipynb
│
├── 3_Duplicate_Removal/
│   └── 1_Duplicate_Removal.ipynb
│
├── 4_Repeated_Character_Normalization/
│   └── 1_Repeated_Character_Normalization.ipynb
│
└── Noise_Handling.md
```

---

## 1. Rare Word Removal

### Definition
**Rare word removal** filters out words that occur very infrequently across a text corpus (e.g., words appearing only once or twice across thousands of documents).

### Why Remove Rare Words?
- **Noise Reduction:** Many rare words are typos, OCR errors, or unique garbage tokens (`"qwertyuiop"`, `"asdf123"`).
- **Dimensionality Reduction:** Significantly reduces vocabulary size and feature matrix sparsity in Bag-of-Words (BoW) and TF-IDF representations.
- **Prevents Overfitting:** Prevents machine learning models from learning spurious associations from single-occurrence terms.

### Implementation Methods
1. **Frequency Thresholding (`collections.Counter`):**
   ```python
   from collections import Counter
   word_counts = Counter(all_words)
   rare_words = set([word for word, count in word_counts.items() if count < 2])
   ```
2. **Document Frequency Thresholding (`scikit-learn`):**
   ```python
   from sklearn.feature_extraction.text import CountVectorizer
   # Ignore terms appearing in fewer than 2 documents
   vectorizer = CountVectorizer(min_df=2)
   ```

---

## 2. Frequent Word Removal

### Definition
**Frequent word removal** filters out words that appear excessively across almost all documents in a specific dataset (beyond standard generic stopword lists).

### Why Remove Highly Frequent Words?
- **Low Information Gain:** Words appearing in $> 80\%$ of documents (e.g., `"patient"` in medical notes or `"court"` in legal documents) provide zero discriminative signal for topic modeling or classification.
- **TF-IDF Optimization:** Prevents domain-specific high-frequency words from drowning out informative keywords.

### Implementation Methods
1. **Scikit-Learn `max_df` Thresholding:**
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   # Ignore terms appearing in more than 85% of documents
   vectorizer = TfidfVectorizer(max_df=0.85)
   ```
2. **Top-K Frequency Removal (`nltk.FreqDist`):**
   ```python
   most_frequent = set([word for word, count in word_counts.most_common(5)])
   ```

---

## 3. Duplicate Removal

### Definition
**Duplicate removal** eliminates duplicate text rows, identical sentences, or near-identical passages from a dataset.

### Why Remove Duplicates?
- **Prevents Data Leakage:** Having the same text sample in both training and test sets produces falsely inflated accuracy metrics.
- **Prevents Model Bias:** Models trained on duplicate items overfit to repeated spam messages or copy-pasted comments.
- **Memory Efficiency:** Reduces overall dataset size and speeds up training pipelines.

### Types of Deduplication
1. **Exact Deduplication:**
   ```python
   df = df.drop_duplicates(subset=['text'])
   ```
2. **Normalized Deduplication:** Lowercasing, stripping punctuation, and removing extra whitespaces before checking uniqueness.
   ```python
   import re
   df['clean'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower().strip()))
   df = df.drop_duplicates(subset=['clean'])
   ```
3. **Fuzzy / Near-Duplicate Detection:** Using `difflib.SequenceMatcher` or MinHash/LSH algorithms for partial string matching.

---

## 4. Repeated Character Normalization

### Definition
**Repeated character normalization** standardizes elongated words containing artificially repeated characters (e.g., `"sooooo"` $\rightarrow$ `"so"`, `"happpyyyy"` $\rightarrow$ `"happy"`, `"cooooool"` $\rightarrow$ `"cool"`).

### Why Normalize Repeated Characters?
- **Social Media & Informal Text Cleaning:** Informal text (tweets, product reviews) frequently uses character repetition for emphasis (`"loveeee"`, `"gooddd"`).
- **Out-of-Vocabulary (OOV) Reduction:** Maps elongated slang words back to standard dictionary vocabulary entries.

### Implementation Methods
1. **Regular Expressions (Regex Back-references):**
   ```python
   import re
   # Replaces characters repeated 3 or more times with a single character
   normalized_text = re.sub(r'(.)\1{2,}', r'\1', text)
   ```
2. **Dictionary-Aware Normalization (NLTK English Dictionary):**
   Checking candidate replacements against `nltk.corpus.words` to preserve legitimate double letters (e.g., preserving `"cool"` while fixing `"cooooool"`).

---

## Summary Comparison Table

| Noise Handling Technique | Target Noise Type | Primary Benefit | Common Tool / Library |
| :--- | :--- | :--- | :--- |
| **Rare Word Removal** | Low-frequency typos / outliers | Reduces matrix sparsity & overfitting | `CountVectorizer(min_df=N)` / `Counter` |
| **Frequent Word Removal** | Corpus-specific hyper-frequent terms | Removes uninformative domain noise | `TfidfVectorizer(max_df=0.85)` |
| **Duplicate Removal** | Identical / near-identical rows | Prevents train/test data leakage & bias | `pandas.drop_duplicates()` |
| **Repeated Character Normalization** | Elongated words (`"sooo"`) | Maps informal text to standard vocabulary | `re.sub(r'(.)\1{2,}', r'\1')` |
