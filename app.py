import os
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import chromadb
from chromadb.utils import embedding_functions
import uuid
import pandas as pd
import time
from transformers import pipeline
from sklearn.metrics import precision_score

# Initialize models and database
@st.cache_resource
def initialize_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline('text-generation', model='distilgpt2', max_new_tokens=50)  # Lightweight LLM
    client = chromadb.PersistentClient(path="./chroma_db")  # Use persistent storage to avoid recreation
    collection = client.get_or_create_collection(
        name="products",
        metadata={"hnsw:space": "cosine"}  # Use cosine for better similarity with SentenceTransformer
    )
    return embedder, generator, collection

# Load and preprocess product data
def load_product_data(file_path='products.json'):
    with open(file_path, 'r') as f:
        products = json.load(f)
    
    # Use product name as ID for upsert to avoid duplicates
    documents = []
    metadatas = []
    ids = []
    
    for product in products:
        # Basic sentiment analysis on reviews
        review_text = " ".join(product['reviews'])
        sentiment = TextBlob(review_text).sentiment.polarity  # Range: -1 (negative) to 1 (positive)
        
        # Combine all text fields for embedding
        text = f"{product['name']} {product['description']} {product['specifications']} {review_text}"
        documents.append(text)
        metadatas.append({
            'name': product['name'],
            'category': product['category'],
            'price': product['price'],
            'sentiment': sentiment,
            'description': product['description'],
            'specifications': product['specifications'],
            'reviews': json.dumps(product['reviews']),  # Store as JSON string for later retrieval
            'image_url': product.get('image_url', '')  # Add image URL
        })
        ids.append(product['name'].replace(" ", "_").lower())  # Deterministic ID based on name
    
    return documents, metadatas, ids, products  # Return full products for later use

# Store data in Chroma
@st.cache_resource
def store_in_chroma(_collection, documents, metadatas, ids, _embedding_fn):
    _collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=_embedding_fn.encode(documents).tolist()
    )
    return _collection


# Retrieve relevant products
def retrieve_products(collection, query, embedder, preferences, top_k=5):
    query_embedding = embedder.encode([query])[0]
    
    # Build where clause for category if specified
    where_clause = {}
    if preferences['category']:
        where_clause['category'] = preferences['category']
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],  # Ensure list
        n_results=top_k,
        where=where_clause if where_clause else None
    )
    
    # Post-filter for price range
    filtered_results = {
        'documents': [[]],
        'metadatas': [[]],
        'distances': [[]],
        'ids': [[]]
    }
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            metadata = results['metadatas'][0][i]
            price = metadata['price']
            if preferences['min_price'] <= price <= preferences['max_price']:
                for key in filtered_results:
                    filtered_results[key][0].append(results[key][0][i])
    
    return filtered_results

# Recommend products based on user preferences
def recommend_products(collection, preferences, embedder, past_queries=[]):
    # Incorporate past queries for basic adaptation
    adapted_query = preferences['query']
    if past_queries:
        adapted_query += " " + " ".join(past_queries[-3:])  # Append last 3 queries for context
    
    results = retrieve_products(collection, adapted_query, embedder, preferences)
    
    # If fewer than 3 results, try without category filter
    if len(results['documents'][0]) < 3 and preferences['category']:
        st.info("Few results found; trying without category filter.")
        preferences_no_cat = preferences.copy()
        preferences_no_cat['category'] = ""
        results = retrieve_products(collection, adapted_query, embedder, preferences_no_cat)
    
    recommendations = []
    for i in range(len(results['documents'][0])):
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        # Normalize cosine distance to similarity (0-1, higher better)
        similarity = 1 - distance  # Since cosine distance = 1 - similarity
        # Weight by sentiment
        weighted_score = similarity * ((metadata['sentiment'] + 1) / 2)
        recommendations.append({
            'id': results['ids'][0][i],
            'name': metadata['name'],
            'category': metadata['category'],
            'price': metadata['price'],
            'score': weighted_score,
            'description': metadata['description'],
            'specifications': metadata['specifications'],
            'reviews': json.loads(metadata['reviews']),
            'sentiment': metadata['sentiment'],
            'image_url': metadata['image_url']
        })
    
    # Sort by weighted score
    recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    return recommendations[:3]

# Generate comparison with LLM
def generate_comparison(recs, generator):
    if len(recs) < 2:
        return None, "Not enough products to compare."
    
    # Create comparison table
    data = []
    for rec in recs:
        data.append({
            'Name': rec['name'],
            'Price': f"${rec['price']}",
            'Sentiment': f"{rec['sentiment']:.2f}",
            'Description': rec['description'],
            'Specifications': rec['specifications'],
            'Reviews Summary': " ".join(rec['reviews'][:2]) + "..."  # Truncated
        })
    
    df = pd.DataFrame(data)
    
    # Generate natural language comparison
    prompt = "Compare the following products based on price, sentiment, and features:\n"
    for rec in recs:
        prompt += f"- {rec['name']}: ${rec['price']}, Sentiment: {rec['sentiment']:.2f}, {rec['description']}\n"
    prompt += "Provide a concise comparison."
    
    try:
        generated = generator(prompt, num_return_sequences=1)[0]['generated_text']
        # Extract just the generated part (remove prompt)
        generated = generated[len(prompt):].strip() or "Comparison generated."
    except Exception as e:
        generated = f"Error generating comparison: {str(e)}"
    
    return df, generated

# Evaluate retrieval precision
def evaluate_retrieval(collection, embedder):
    # Simulated test cases: query -> expected product IDs (update based on your products.json IDs)
    test_cases = [
        {'query': 'wireless headphones with long battery life', 'expected': ['wireless_headphones']},
        {'query': 'comfortable running shoes', 'expected': ['running_shoes']},
        {'query': 'python programming guide', 'expected': ['python_programming_book']},
        {'query': 'latest smartphone with camera', 'expected': ['smartphone']},
        {'query': 'warm winter jacket', 'expected': ['winter_jacket']}
    ]
    
    y_true = []
    y_pred = []
    latencies = []
    
    for test in test_cases:
        start_time = time.time()
        results = retrieve_products(collection, test['query'], embedder, 
                                    preferences={'category': '', 'min_price': 0, 'max_price': 1000})
        latency = time.time() - start_time
        latencies.append(latency)
        
        retrieved_ids = results['ids'][0] if results['ids'] and results['ids'][0] else []
        # True if any expected ID is in top retrieved
        y_true.append(1)
        y_pred.append(1 if any(exp in retrieved_ids for exp in test['expected']) else 0)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    return precision, avg_latency

# Streamlit UI
def main():
    st.title("E-commerce Product Recommender")
    st.write("Enter your preferences to get personalized product recommendations.")

    # Initialize models
    embedder, generator, collection = initialize_models()

    # Load and store data
    documents, metadatas, ids, full_products = load_product_data()
    collection = store_in_chroma(collection, documents, metadatas, ids, embedder)

    # Session state for user adaptation
    if 'past_queries' not in st.session_state:
        st.session_state.past_queries = []
    if 'liked_products' not in st.session_state:
        st.session_state.liked_products = []

    # User input
    query = st.text_input("What are you looking for? (e.g., 'wireless headphones with long battery life')")
    category = st.selectbox("Category", ["All", "Electronics", "Clothing", "Books"])
    min_price = st.number_input("Minimum Price", min_value=0.0, value=0.0)
    max_price = st.number_input("Maximum Price", min_value=0.0, value=1000.0)

    if st.button("Get Recommendations"):
        if query:
            preferences = {
                'query': query,
                'category': category if category != "All" else "",
                'min_price': min_price,
                'max_price': max_price
            }
            recommendations = recommend_products(collection, preferences, embedder, st.session_state.past_queries)
            
            # Update session state
            st.session_state.past_queries.append(query)
            
            st.subheader("Recommended Products")
            for rec in recommendations:
                st.write(f"**{rec['name']}** (Category: {rec['category']}, Price: ${rec['price']})")
                if rec['image_url']:
                    st.image(rec['image_url'], caption=rec['name'], width=200)
                st.write(f"Recommendation Score: {rec['score']:.2f}")
                st.write(f"Description: {rec['description']}")
                st.write("Reviews: " + ", ".join(rec['reviews']))
                if st.button(f"Like {rec['name']}", key=rec['id']):
                    st.session_state.liked_products.append(rec['name'])
                    st.write("Added to likes!")
                st.write("---")
            
            # Comparison feature
            if len(recommendations) > 1:
                st.subheader("Product Comparison")
                comparison_df, comparison_text = generate_comparison(recommendations, generator)
                if comparison_df is not None:
                    st.dataframe(comparison_df)
                st.write("**Generated Comparison Summary**:")
                st.write(comparison_text)
            
            # Basic evaluation metrics
            start_time = time.time()
            retrieve_products(collection, query, embedder, preferences)
            latency = time.time() - start_time
            st.write(f"Retrieval Latency: {latency:.2f} seconds")
            
            # Simulated accuracy: Assume top recommendation matches query (dummy metric)
            accuracy = len(recommendations) / 3 if recommendations else 0
            st.write(f"Simulated Retrieval Coverage: {accuracy:.2f}")
            
            # Advanced evaluation
            precision, avg_latency = evaluate_retrieval(collection, embedder)
            st.write(f"Retrieval Precision (Simulated): {precision:.2f}")
            st.write(f"Average Retrieval Latency (Simulated): {avg_latency:.2f} seconds")
        else:
            st.error("Please enter a search query.")

    # Display liked products for adaptation
    if st.session_state.liked_products:
        st.sidebar.subheader("Liked Products")
        st.sidebar.write(", ".join(st.session_state.liked_products))

if __name__ == "__main__":
    main()