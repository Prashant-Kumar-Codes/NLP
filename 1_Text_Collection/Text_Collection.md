# 1. Text Collection in NLP

Text Collection is the foundational step in any Natural Language Processing (NLP) pipeline. Before text can be cleaned, preprocessed, or fed into machine learning / deep learning models, raw textual data must be gathered from various data sources.

---

## Folder Structure Overview

```
1_Text_Collection/
├── 1_Reading_Text_Files/
│   └── 1_Reading_Text_Files.ipynb
|
├── 2_Reading_CSV_Excel/
│   └── 1_Reading_CSV_Excel.ipynb
|
├── 3_Reading_JSON_XML/
│   └── 1_Reading_JSON_XML.ipynb
|
├── 4_Web_Scraping/
│   └── 1_Web_Scraping.ipynb
|
├── 5_APIs/
│   └── 1_APIs.ipynb
|
├── 6_Database_Extraction/
│   └── 1_Database_Extraction.ipynb
|
├── 7_Building_Corpus/
│   └── 1_Building_Corpus.ipynb
|
└── 8_Train_Val_Test_Split/
    └── 1_Train_Val_Test_Split.ipynb
```

---

## 1. Text Collection Methods

### 1. Reading Text Files (`.txt`)
- Standard text files stored locally or on cloud storage.
- **Key Concepts:** Encoding management (`utf-8`, `latin-1`, `cp1252`), line-by-line streaming for large files (`for line in open(...)`), reading multiple documents with `pathlib` / `glob`.

### 2. Reading CSV / Excel Files (`.csv`, `.tsv`, `.xlsx`)
- Tabular data formats containing text columns alongside metadata and target labels.
- **Key Concepts:** Python `csv` module, `pandas.read_csv`, `pandas.read_excel`, handling delimiters, handling missing text values (`fillna`, `dropna`).

### 3. Reading JSON & XML Files (`.json`, `.jsonl`, `.xml`)
- Semi-structured data formats. JSON Lines (`.jsonl`) is the industry standard for LLM pre-training and fine-tuning datasets.
- **Key Concepts:** `json.load`, `pandas.read_json(..., lines=True)`, `pandas.json_normalize` for nested structures, `xml.etree.ElementTree` / `BeautifulSoup` for XML parsing.

### 4. Web Scraping
- Harvesting raw unstructured text from websites, blogs, news feeds, or forums.
- **Key Concepts:** HTTP GET requests (`requests`), HTML parsing (`bs4` / `BeautifulSoup`), stripping HTML tags/scripts/styles, User-Agent headers, rate limiting.

### 5. API Text Extraction
- Fetching text dynamically from third-party RESTful APIs (e.g., Wikipedia API, Twitter/X API, News API).
- **Key Concepts:** REST endpoints, authentication (API Keys, Bearer tokens), request parameters, handling API status codes & JSON paylods.

### 6. Database Extraction
- Retrieving text columns from Relational (SQLite, PostgreSQL, MySQL) or NoSQL (MongoDB) databases.
- **Key Concepts:** SQL `SELECT` queries, `sqlite3`, loading query results directly into Pandas DataFrames via `pd.read_sql_query`, chunked batch processing for large databases.

### 7. Building a Corpus
- Combining and aggregating text gathered from multiple heterogeneous sources into a unified, standardized corpus.
- **Key Concepts:** Schema standardization (`doc_id`, `text`, `source`, `timestamp`), document metadata tagging, vocabulary summary stats, exporting to `.jsonl` / `.parquet`.

### 8. Train / Validation / Test Split
- Splitting the raw or preprocessed dataset into distinct subsets to prepare for model training and evaluation.
- **Key Concepts:** `scikit-learn` `train_test_split`, **Stratified Splitting** to maintain label distribution balance, preventing data leakage across splits.
