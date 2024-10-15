
# Sentence Embeddings and Similarity Search with Ollama and FAISS

## Project Overview
This project demonstrates how to convert sentences into embeddings using the Ollama model (Llama2) and store them in a vector database (FAISS). The system can then perform a similarity search to find the most semantically similar sentence from a collection.

## Features
- Convert sentences into embeddings using Ollama.
- Store embeddings in FAISS for efficient similarity search.
- Perform similarity search to find the closest match to a given query.

## Requirements
To run this project, you'll need the following:
- Python 3.8+
- `ollama`
- `faiss`
- `numpy`

You can install the required libraries using the following command:
```bash
pip install ollama faiss numpy
```

## Usage
1. **Convert Sentences to Embeddings**  
   The script converts a set of sample sentences into embeddings using Ollama and stores them in FAISS.

2. **Perform Similarity Search**  
   After storing the embeddings, you can input a new sentence, and the system will return the most similar sentence from the stored collection.

### Example
```python
import faiss
import ollama
import numpy as np

def text_to_embedding_ollama(text):
    model_name = "llama2"
    response = ollama.embed(model=model_name, input=text)
    return response

# Sample sentences
sentences = [
    "Artificial Intelligence is the future.",
    "AI requires large datasets to train models.",
    "Machines learn by analyzing data."
]

# Convert to embeddings
embeddings = [text_to_embedding_ollama(sentence) for sentence in sentences]

# Store in FAISS index for similarity search
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Query the most similar sentence
query = text_to_embedding_ollama("AI evolves with data.")
D, I = index.search(np.array([query]), 1)
print(f"The most similar sentence is: {sentences[I[0][0]]}")
```

## How It Works
1. **Embeddings Generation**: Each sentence is converted into an embedding using the Ollama model, which outputs a high-dimensional vector representation.
2. **FAISS Vector Search**: The embeddings are stored in FAISS, a vector search library optimized for fast similarity searches. It uses the L2 distance (Euclidean) to determine the most similar sentence to the input query.

## Testing the System
You can test the system by adding new sentences and running similarity searches to verify the correctness of the results.

## License
This project is licensed under the MIT License.
