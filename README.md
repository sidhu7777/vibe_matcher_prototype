---
# **Vibe Matcher Prototype**

### **Project Overview**

The **Vibe Matcher** is a prototype recommendation system built to demonstrate how AI can enable *vibe-based product discovery*.
Instead of relying on keywords, it understands the *semantic meaning* behind a user's vibe query (e.g., “boho summer” or “urban chic”) and recommends the top-3 most relevant fashion products based on their descriptions and style tags.

This project uses **OpenAI’s `text-embedding-ada-002` model** to convert text into high-dimensional vectors and computes similarity using **cosine similarity** via `scikit-learn`.
---

.

---

## **Project Workflow**

1. **Data Preparation**

   - Created a `pandas` DataFrame with **10 mock fashion products**, each having a `name`, `description`, and `tags` list.
   - Combined fields (`name + desc + tags`) to enrich semantic representation before embedding.

2. **Embeddings**

   - Generated embeddings using **OpenAI’s text-embedding-ada-002 model**.
   - Stored embeddings directly in the DataFrame for vector similarity computation.

3. **Vector Search (Cosine Similarity)**

   - Used `sklearn.metrics.pairwise.cosine_similarity` to measure similarity between the user’s vibe query embedding and each product’s embedding.
   - Ranked the **top-3 most similar items** for each query.

4. **Fallback Handling**

   - Added logic to detect when the top similarity score falls below 0.7.
   - Displays a friendly message suggesting users try a different vibe query when no strong match is found.

5. **Evaluation & Metrics**

   - Tested with four queries (three in-domain, one out-of-domain).
   - Logged per-query latency and similarity scores.
   - Added a `good_in_top3` column showing how many of the top-3 results exceeded the similarity threshold (0.7).
   - Plotted latency for all queries using Matplotlib.

---

## **Evaluation Summary**

| Query                                   | Latency (s) | Top-1 Score | Good in Top-3 | Good Match |
| --------------------------------------- | ----------- | ----------- | ------------- | ---------- |
| boho summer                             | 2.45        | 0.893       | 3/3           | True       |
| minimalist cozy                         | 0.44        | 0.891       | 3/3           | True       |
| energetic urban chic                    | 0.75        | 0.856       | 3/3           | True       |
| quantum computer processor architecture | 0.46        | 0.678       | 0/3           | False      |

**Result:**
3 out of 4 queries met the similarity threshold (> 0.7).
The intentionally out-of-domain query correctly triggered the fallback, proving robust edge-case handling.

---

## **Reflection**

- Successfully built an end-to-end vibe-based recommendation prototype using OpenAI embeddings.
- Implemented **data preprocessing improvements** by combining product names, descriptions, and tags for more context-rich embeddings.
- Added evaluation metrics including latency tracking, top-3 quality counts, and fallback handling for low-similarity queries.
- Observed that semantically similar fashion items cluster closely, resulting in consistently high similarity scores.
- Future improvements:

  - Integrate **Pinecone or FAISS** for scalable vector storage and retrieval.
  - Add **multimodal embeddings** (text + image) for deeper fashion understanding.
  - Expand dataset diversity to improve differentiation across product types.
  - Introduce a learned re-ranker for personalized retrieval refinement.

---

## **Setup Instructions**

### 1. **Clone Repository or Download Notebook**

```bash
git clone https://github.com/sidhu7777/vibe_matcher_prototype.git
cd vibe-matcher-prototype
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Add OpenAI API Key**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

### 4. **Run the Notebook**

Open `vibe_matcher_prototype.ipynb` in **VS Code** or **Jupyter Notebook**, then run all cells:

```bash
jupyter notebook vibe_matcher_prototype.ipynb
```

---

## **File Structure**

```
vibe_matcher_prototype/
│
├── vibe_matcher_prototype.ipynb   # Main notebook (complete implementation)
├── requirements.txt                # Python dependencies
├── .env                            # OpenAI API key (not shared)
├── .gitignore                      # Ignore sensitive files and checkpoints
└── README.md                       # Project overview and documentation
```

---

## **Technologies Used**

- **Python 3.10+**
- **OpenAI API (text-embedding-ada-002)**
- **Pandas** — Data handling
- **NumPy** — Array operations
- **scikit-learn** — Cosine similarity
- **Matplotlib** — Latency visualization
- **python-dotenv** — Environment variable management

---

## **Acknowledgments**

This project was developed as part of the Nexora AI challenge to demonstrate practical understanding of embeddings, similarity search, and evaluation methodology in recommendation systems.

---

## **Author**

**[Vineeth Raja Banala]**
AI Developer / Data Science Enthusiast
_Email:_ [[bvineeth76@gmail.com]]

---
