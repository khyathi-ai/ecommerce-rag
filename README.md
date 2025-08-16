# E-commerce Product Recommendation RAG System

A Retrieval-Augmented Generation (RAG) system for personalized e-commerce product recommendations, built with Streamlit, SentenceTransformers, ChromaDB, and TextBlob.

## Features
- Combines product descriptions, specifications, and reviews for semantic search.
- Sentiment analysis on reviews using TextBlob.
- Personalized recommendations with category and price filtering.
- Side-by-side product comparison table and generated summaries using distilgpt2.
- Basic user preference adaptation via session state.
- Evaluation metrics: retrieval precision and latency.

## Setup
1. Clone the repo: `git clone https://github.com/khyathi-ai/ecommerce-rag.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `streamlit run app.py`

## Architecture
- **Embedding**: SentenceTransformer (`all-MiniLM-L6-v2`) for text embeddings.
- **Storage/Retrieval**: ChromaDB with cosine similarity.
- **Generation**: distilgpt2 for comparison summaries.
- **UI**: Streamlit for interactive input and display.
- [Optional: Add architecture_diagram.png]

## Limitations
- Limited to text data; future work could include image embeddings.
- Basic user adaptation; no long-term user profiles.
- Small product dataset; scalable with more data.

## Evaluation
- Precision: ~0.67 (based on simulated queries).
- Avg. Latency: ~0.1-0.3s for retrieval.

## Future Improvements
- Multimodal support (image embeddings).
- Advanced user adaptation via feedback loops.
- Integration with larger LLMs for better generation.
