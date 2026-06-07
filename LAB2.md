
## Lab 2: Storing Embeddings in a Vector Database

In Lab 1, you generated embeddings and calculated similarity scores. But what if you have hundreds or thousands of sentences? You need a way to store them and search through them efficiently. That's where **vector databases** come in.

In this lab, you'll use **LangChain's In-Memory Vector Store** - a simple vector database that works great for learning and small-scale applications. You'll store sentences along with their embeddings, then build a CLI search feature that finds the most similar sentences to a user's query.

#### What is the In-Memory Vector Store?

LangChain's In-Memory Vector Store is a lightweight vector database that stores all data in memory while your program runs. It's perfect for:
- Learning and experimentation
- Testing vector search functionality
- Prototyping RAG applications
- Small datasets that can fit in memory

For production applications, you can use persistent vector databases like Chroma, Pinecone, or Qdrant, but the concepts are the same. Remember to run your code after each step. You want to incrementally make changes and test so you can be a good AI pair programmer!

---

### Step 1: Import Vector Store

Before storing data in a vector database, you need to import the In-Memory Vector Store from LangChain.

**Prompt 1: Import InMemoryVectorStore**
```
Import InMemoryVectorStore from langchain_core.vectorstores at the top of app.py.
Also import datetime for timestamp metadata.
```

**What to Look For:** The InMemoryVectorStore should be imported. In Python with LangChain, you don't need to define a separate data model class - the vector store handles the structure automatically. Each document will have:
- A unique ID (auto-generated or specified)
- The original sentence text (stored as document content)
- The embedding vector (generated automatically)
- Metadata (stored as a dictionary)

---

### Step 2: Initialize the Vector Store

Now you need to create an instance of InMemoryVectorStore with your embedding model.

**📚 Required Documentation:**
- [LangChain InMemoryVectorStore](https://docs.langchain.com/oss/javascript/integrations/vectorstores/memory)

**Prompt 2: Create InMemoryVectorStore**
```
After initializing the embeddings model in main(), create an InMemoryVectorStore instance:
- Create it with InMemoryVectorStore(embeddings) where embeddings is your OpenAIEmbeddings instance
- Store it in a variable called vector_store
```

**What to Look For:** The code should create an in-memory vector store instance that you can add documents to and search through.

---

### Step 3: Store Sentences with Their Embeddings

Instead of just printing the embeddings like in Lab 1, you'll now store them in the vector database.

**Prompt 3: Store Sentences in Vector Database**
```
Replace the current embedding generation code with the following logic:
1. Create the sentences list with 3 test sentences (same as Lab 1)
2. Use the vector_store.add_texts() method to add all sentences at once
3. Add metadata for each sentence including:
   - created_at: Current timestamp using datetime.now().isoformat()
   - index: The position in the original list
4. Print a confirmation message showing how many sentences were stored
5. Print each sentence that was added
```

**What to Look For:** This should replace the code that just calculated similarities. Now you're persisting the data in the vector store for later retrieval. The InMemoryVectorStore automatically generates and stores the embeddings.

---

### Step 4: Build a Semantic Search Function

Now comes the powerful part - searching! You'll create a function that takes a user's query and finds the most similar sentences in the database.

**📚 Reference:** [LangChain Vector Store Retrieval](https://docs.langchain.com/oss/javascript/langchain/retrieval) - Learn about vector similarity search methods.

**Prompt 4: Create a Semantic Search Function**
```
Create a function called search_sentences that:
- Takes parameters: the vector_store and a search query string
- Optionally takes k (default to 3) for the number of results to return
- Uses vector_store.similarity_search_with_score() to find similar documents
- Returns the top k most similar results with their similarity scores
- Print the results with formatting showing:
  - Rank number (1, 2, 3...)
  - Similarity score formatted to 4 decimal places
  - The sentence text
```

**What to Look For:** GitHub Copilot should create a function that performs vector similarity search and displays results ranked by relevance.

---

### Step 5: Create an Interactive CLI Search Loop

Finally, make your application interactive so users can type queries and see results in real-time.

**Prompt 5: Add Interactive Search CLI**
```
After storing the sentences, add an interactive loop that:
1. Display a header "=== Semantic Search ==="
2. Start a while True loop
3. Prompt the user to "Enter a search query (or 'quit' to exit): " using input()
4. If the user types "quit" or "exit", break the loop
5. If the input is empty or whitespace only, continue to the next iteration
6. Otherwise, call the search_sentences function with the user's query
7. Display the results
8. Print a blank line between searches for readability
9. After the loop ends, print a goodbye message
```

### Step 6: Test Your Semantic Search

**Test Point**: Run the application using `python app.py`. You should see:

1. Confirmation that sentences were stored in the vector database
2. A prompt asking you to enter a search query
3. Ability to type queries like:
   - "animal sounds" (should find sentences about dogs barking)
   - "physics" (should find the electron sentence)
   - "pet making noise" (should find dog-related sentences)
   - "scientific particles" (should find the electron sentence)

**Try These Queries:**
- "puppy barking" - Should match the dog sentences even though "puppy" wasn't in the original text
- "subatomic particles rotating" - Should match the electron sentence
- "cat meowing" - Should still match dog sentences (both are animals making sounds)

**What You'll Notice:**
- You don't need exact keyword matches
- The search understands semantic meaning
- Results are ranked by similarity (closest matches first)
- Related concepts cluster together even with different wording

---

### Step 7: Expand Your Dataset

Now that you have a working vector search system, add more sentences to make it more interesting!

**Prompt 6: Add More Diverse Sentences**
```
Expand the sentences list to include at least 10-15 sentences covering different topics such as:
- Animals and pets
- Science and physics
- Food and cooking
- Sports and activities
- Weather and nature
- Technology and programming

Keep the sentences varied so you can test how well semantic search groups related concepts.
```

**Example Dataset You Might Create:**
```python
sentences = [
    "The canine barked loudly.",
    "The dog made a noise.",
    "The electron spins rapidly.",
    "I love eating pizza with extra cheese.",
    "The basketball player scored a three-pointer.",
    "Rain is forecasted for tomorrow afternoon.",
    "Python is a popular programming language.",
    "The kitten purred softly on the couch.",
    "Quantum mechanics explains particle behavior.",
    "Homemade pasta tastes better than store-bought.",
    "The soccer match ended in a tie.",
    "Clouds are forming over the mountains.",
    "JavaScript runs in web browsers.",
    "Puppies need lots of attention and exercise.",
    "Atoms are made of protons, neutrons, and electrons."
]
```

**Test Again**: Run the application and try queries like:
- "pets" - Should find dog, cat, puppy, and kitten sentences
- "science" - Should find physics, quantum mechanics, and atom sentences
- "Italian food" - Should find pizza and pasta sentences
- "code" - Should find programming language sentences

---

### Expected Output

```
🤖 Python LangChain Agent Starting...

=== Vector Store Lab ===

Storing 15 sentences in the vector database...
✅ Successfully stored 15 sentences

=== Semantic Search ===

Enter a search query (or 'quit' to exit): pets

🔍 Search Results for "pets":

1. [Score: 0.XXXX] The canine barked loudly.
2. [Score: 0.XXXX] The dog made a noise.
3. [Score: 0.XXXX] Puppies need lots of attention and exercise.

Enter a search query (or 'quit' to exit): science

🔍 Search Results for "science":

1. [Score: 0.XXXX] Quantum mechanics explains particle behavior.
2. [Score: 0.XXXX] Atoms are made of protons, neutrons, and electrons.
3. [Score: 0.XXXX] The electron spins rapidly.

Enter a search query (or 'quit' to exit): quit

👋 Goodbye!
```

---

### Discussion Questions

After completing Lab 2, consider:

1. **Keyword vs. Semantic Search**: How does vector search differ from traditional keyword search? What advantages does it offer?

2. **Similarity Scores**: Why do some queries return higher similarity scores than others? What factors affect the score?

3. **False Positives**: Did any of your queries return unexpected results? Why might that happen?

4. **Scaling Considerations**: The in-memory Chroma store keeps everything in memory. What challenges would arise with 1 million sentences? 

5. **Real-World Applications**: Where would semantic search be useful in applications you use daily (search engines, customer support, content recommendations)?

---

### Extension Challenges

If you finish early or want to go deeper:

**Challenge 1: Add Metadata Filtering**
```
Modify the metadata dictionary to include a "category" key (e.g., "animals", "science", "food").
Update the search to allow filtering by category using metadata filtering.
Ask Copilot: "Add category-based filtering to the vector search to only search within specific categories using metadata filters"
```

**Challenge 2: Hybrid Search**
```
Combine vector similarity search with keyword matching.
Ask Copilot: "Implement hybrid search that combines vector similarity with keyword matching for better precision"
```

**Challenge 3: Threshold Filtering**
```
Only show results above a minimum similarity score (e.g., 0.7).
Ask Copilot: "Add a similarity threshold parameter to filter out low-confidence results"
```

**Challenge 4: Persist to Disk**
```
Replace InMemoryVectorStore with a persistent vector store like Chroma or FAISS.
Add the ability to save and reload the vector store from disk.
Ask Copilot: "Replace InMemoryVectorStore with Chroma using persist_directory to save data to disk"
```

---
