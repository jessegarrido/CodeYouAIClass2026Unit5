## Lab 1: Embedding Inspector

This lab introduces you to **text embeddings**, which are numerical representations of text that AI models use to understand semantic meaning.

#### Setup: Deploy the Embedding Model
1. Visit https://github.com/marketplace/models
2. Search for "text-embedding-3-small" in the models list
3. Click on the model and select "Get started" or "Deploy" to enable it
4. The model will now be available through your GitHub token (the same one you're using for chat)

#### Understanding Vector Similarity

**What are Embeddings?**

When an AI model processes text, it converts words into numbers. But instead of just counting words, it creates something called an **embedding** - a list of numbers (a vector) that captures the *meaning* of the text.

Think of it this way:
- Each sentence becomes a point in a multi-dimensional space (like coordinates on a map)
- Sentences with similar meanings are placed close together
- Sentences with different meanings are placed far apart

**What is Cosine Similarity?**

To measure how close two sentences are in meaning, we calculate the **cosine similarity** between their embedding vectors:
- A score of **1.0** means the sentences are very similar (almost identical in meaning)
- A score of **0.0** means they're completely unrelated
- A score of **-1.0** means they're opposite in meaning

It's like measuring the angle between two arrows:
- If the arrows point in the same direction → high similarity (close to 1)
- If the arrows point in different directions → low similarity (close to 0)

#### Build the Embedding Inspector

**Documentation Reference:**
- [LangChain OpenAI Embeddings](https://python.langchain.com/docs/integrations/text_embedding/openai/)
- [LangChain Embeddings Overview](https://python.langchain.com/docs/concepts/embedding_models/)

> **Note:** The starter code already includes imports, a `cosine_similarity()` helper function, and basic environment setup. You'll add the embedding inspector functionality to the `main()` function.

**Prompt 1: Initialize Embedding Model**
```
In the main() function, after the GITHUB_TOKEN check, create an OpenAIEmbeddings instance:
- model="text-embedding-3-small"
- base_url="https://models.inference.ai.azure.com" (for GitHub Models API)
- api_key=os.getenv("GITHUB_TOKEN")

NOTE: Set check_embedding_ctx_length=False as we need to demonstrate failures
```

**Prompt 2: Create Test Sentences**
```
Create a list of three test sentences:
1. "The canine barked loudly."
2. "The dog made a noise."
3. "The electron spins rapidly."

Add print statements for:
- "=== Embedding Inspector Lab ==="
- "Generating embeddings for three sentences..."
```

**Prompt 3: Generate Embeddings**
```
Create a loop to:
- Generate an embedding for each sentence using embeddings.embed_query()
- Store each embedding vector in a list
- Print each sentence with a number (Sentence 1, Sentence 2, etc.)
```

**Prompt 4: Calculate Cosine Similarities**
```
Add code to calculate and display the cosine similarity between:
- Sentence 1 and Sentence 2
- Sentence 2 and Sentence 3
- Sentence 3 and Sentence 1

Use the cosine_similarity() function (already provided in the starter code)
Format the output to 4 decimal places using f-strings with :.4f
Display results with clear labels showing which sentences are being compared
```

> **Note:** You can create a `cosine_similarity` helper function with a single prompt: "Write a function to calculate cosine similarity using the math module." If you'd like to practice, try generating it yourself!

**Test Point**: Run the application using `python app.py` or the Python run configuration. You should see:
- Each sentence printed
- Cosine similarity scores between each pair
- Sentences 1 and 2 (both about dogs) will have a HIGH similarity (around 0.7-0.9)
- Sentence 3 (about electrons) will have LOW similarity with sentences 1 and 2 (around 0.1-0.3)

#### Expected Output

```
🤖 Python LangChain Agent Starting...

=== Embedding Inspector Lab ===

Generating embeddings for three sentences...

Sentence 1: "The canine barked loudly."
Sentence 2: "The dog made a noise."
Sentence 3: "The electron spins rapidly."

=== Cosine Similarities ===

Cosine similarity between Sentence 1 and Sentence 2: 0.XXXX
Cosine similarity between Sentence 2 and Sentence 3: 0.XXXX
Cosine similarity between Sentence 3 and Sentence 1: 0.XXXX
```

Write down these similarities. What observations do you see?

#### Experiment: Modify Sentences to See How Similarity Changes

Now that you understand the basics, try modifying the sentences to see how the similarity scores change! Here are several experiments you can try:

**Experiment 1: Same Sentence in Different Languages**
```python
sentences = [
    "The dog barked loudly.",
    "El perro ladró fuerte.",  # Spanish
    "Le chien a aboyé fort."   # French
]
```
**What to expect:** High similarity scores! Embeddings understand meaning across languages.

**Experiment 2: Change Just One Word**
```python
sentences = [
    "The cat sat on the mat.",
    "The dog sat on the mat.",
    "The cat sat on the hat."
]
```
**What to expect:** Very high similarity (around 0.9+) because most words are identical.

**Experiment 3: Add Adjectives and Details**
```python
sentences = [
    "I like pizza.",
    "I really love eating delicious, hot pizza.",
    "Pizza is good."
]
```
**What to expect:** Still high similarity - the core meaning (liking pizza) is the same.

**Experiment 4: Opposite Meanings**
```python
sentences = [
    "The movie was excellent and entertaining.",
    "The movie was terrible and boring.",
    "I enjoyed watching the film."
]
```
**What to expect:** Sentences 1 and 2 will have moderate similarity (same topic: movies) but lower than 1 and 3.

**Experiment 5: Technical vs. Casual Language**
```python
sentences = [
    "The precipitation is expected to commence shortly.",
    "It's going to rain soon.",
    "The weather forecast indicates imminent rainfall."
]
```
**What to expect:** High similarity - all mean the same thing despite different vocabulary.

**Experiment 6: Different Topics**
```python
sentences = [
    "I love programming in Python.",
    "Chocolate cake is delicious.",
    "The ocean is very deep."
]
```
**What to expect:** Low similarity across all pairs - completely different topics.

**Your Turn!**

Choose 2-3 experiments above and modify the `sentences` list in your app.py. Run the program each time and observe how the similarity scores change. Write down:
1. Which experiment you tried
2. What the similarity scores were
3. Whether the results matched your expectations

**Discussion Questions:**
- Why do you think sentences in different languages can have high similarity?
- What happens to similarity when you keep the topic the same but change the sentiment (positive vs. negative)?
- How might these embeddings be useful in a real application (search engines, recommendation systems, etc.)?
