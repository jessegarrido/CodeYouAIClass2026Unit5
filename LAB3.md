
## Lab 3: Loading Real Documents - Discovering the Limits

In Labs 1 and 2, you worked with short sentences. But in the real world, you need to search through actual documents like employee handbooks, product manuals, or knowledge bases. In this lab, you'll attempt to load real markdown files into your vector database and discover firsthand why **chunking** is essential.

**📚 Key Concepts:**
- [Token Limits in Embedding Models](https://platform.openai.com/docs/guides/embeddings) - Understanding model limitations
- [LangChain Document Loaders](https://python.langchain.com/v0.2/docs/how_to/#document-loaders) - Best practices for loading data

### The Challenge: From Sentences to Documents

Your workspace contains two markdown files:
- **HealthInsuranceBrochure.md** - A moderate-sized document (~123 lines)
- **EmployeeHandbook.md** - A large document (~3,267 lines)

You'll start by trying to store these documents as-is (without chunking) and observe what happens when you hit the limits of embedding models.

---

### Step 1: Understand the Document Structure

With LangChain and InMemoryVectorStore, you don't need to manually define a data model. Instead, documents are stored with:
- **page_content**: The actual document text
- **metadata**: A dictionary containing information like filename, creation time, etc.
- **id**: A unique identifier (auto-generated or specified)
- **embedding**: The vector representation (generated automatically)

**Prompt 1: Review the Document Structure**
```
I need to understand how to store documents in the InMemoryVectorStore.
Each document will have:
- page_content: the document text
- metadata: a dictionary with 'fileName', 'createdAt', and any other relevant fields
- The embedding is generated automatically by the vector store
```

**What to Look For:** LangChain's Document structure is simpler than Java - no annotations or class definitions needed.

---

### Step 2: Clear Previous Data and Prepare for Documents

Before loading documents, you need to clean up the sentence data from Lab 2.

**Prompt 2: Remove the Sentence Loading Code**
```
Remove all the code that:
- Creates the sentences list
- Adds sentences to the vector store using add_texts()
- The search loop from Lab 2

Keep the vector store initialization code (InMemoryVectorStore), but we'll be adding different data loading logic.
```

**What to Look For:** Your app.py should still initialize the embeddings and vector store, but no longer populate it with sentences.

---

### Step 3: Load the Health Insurance Brochure

Now you'll create a function to read a markdown file and store it in the vector database.

**📚 Reference:** [Python File I/O](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files) - Learn about reading files in Python.

**Prompt 3: Create File Loading Function**
```
Create a function called load_document that:
- Takes parameters: the vector_store and a file_path (string)
- Reads all text from the file using open() and read()
- Creates a LangChain Document object with:
  - page_content: The full document text
  - metadata: A dictionary with:
    - 'fileName': The file name from the path (use os.path.basename(file_path))
    - 'createdAt': Current timestamp using datetime.now().isoformat()
- Adds the document to the vector store using vector_store.add_documents([document])
- Prints a success message with the filename and content length
- Returns the document ID

Add error handling using try-except for FileNotFoundError and other exceptions.
Import Document from langchain_core.documents
```

**What to Look For:** This function encapsulates the logic for loading a single document into the vector database.

---

### Step 4: Load the First Document

Let's try loading the health insurance brochure first.

**Prompt 4: Load HealthInsuranceBrochure.md**
```
After the vector store is created, call the load_document function to load the file "HealthInsuranceBrochure.md" from the workspace root directory.
Display a header "=== Loading Documents into Vector Database ===" before loading.
Print the filename and a confirmation when it loads successfully.
```

**Test Point**: Run the application using `python app.py`. 

**Expected Output:**
```
🤖 Python LangChain Agent Starting...

=== Loading Documents into Vector Database ===

Loading HealthInsuranceBrochure.md...
✅ Successfully loaded HealthInsuranceBrochure.md (... characters)
```

**What You Should Observe:**
- The brochure loads successfully
- No errors occur
- The embedding is generated without issues

This works because the health insurance brochure is small enough to fit within the embedding model's token limits.

---

### Step 5: Try Loading the Employee Handbook

Now let's see what happens when you try to load a much larger document.

**Prompt 5: Add Loading for EmployeeHandbook.md**
```
After loading the HealthInsuranceBrochure.md, add another call to load_document to load "EmployeeHandbook.md" from the workspace root directory.
```

**Test Point**: Run the application again using `python app.py`.

**What Will Happen:**
You'll encounter an error! The embedding model has a maximum token limit (typically around 8,191 tokens for text-embedding-3-small), and the Employee Handbook is too large to process in one go.

**Expected Error:**
```
❌ Error loading EmployeeHandbook.md: 
This model's maximum context length is 8191 tokens. However, your messages resulted in 15234 tokens.
```

**Why This Happens:**
- The EmployeeHandbook.md contains over 3,000 lines of text
- When converted to tokens, it exceeds the embedding model's 8,191 token limit
- The embedding API rejects the request
- Your document cannot be stored as-is

---

### Step 6: Observe and Document the Problem

Before moving to the solution, let's understand what just happened.

**Prompt 6: Add Error Analysis**
```
In the load_document function's except block, enhance the error handling to:
- Check if the exception message (str(e)) contains "maximum context length" or "token"
- If so, display a special message explaining:
  "⚠️ This document is too large to embed as a single chunk."
  "Token limit exceeded. The embedding model can only process up to 8,191 tokens at once."
  "Solution: The document needs to be split into smaller chunks."
- Otherwise, display the regular error message using str(e)
```

**Test Again**: Run `python app.py` one more time.

**Expected Output:**
```
🤖 Python LangChain Agent Starting...

=== Loading Documents into Vector Database ===

Loading HealthInsuranceBrochure.md...
✅ Successfully loaded HealthInsuranceBrochure.md (2,456 characters)

Loading EmployeeHandbook.md...
❌ Error loading EmployeeHandbook.md
⚠️ This document is too large to embed as a single chunk.
Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.
Solution: The document needs to be split into smaller chunks.
```

---

### Understanding the Problem

**📚 Learn More:** [LangChain Text Splitters](https://python.langchain.com/v0.2/docs/how_to/#text-splitters) - Understanding document chunking strategies.

**What You've Discovered:**

1. **Token Limits Are Real** - Embedding models cannot process unlimited text. They have hard limits (typically 512 to 8,191 tokens depending on the model).

2. **Documents Vary in Size** - Some documents fit within limits (HealthInsuranceBrochure.md), while others don't (EmployeeHandbook.md).

3. **One-Size-Fits-All Doesn't Work** - You can't just load every document as a single vector. Large documents must be broken down.

4. **Chunking is Essential** - To work with real-world documents in RAG systems, you need a strategy to divide them into processable pieces.

**The Core Problem:**
```
Full Document (15,234 tokens) → ❌ Too large for embedding model
Full Document Split into Chunks (10 chunks × ~1,500 tokens) → ✅ Each chunk can be embedded
```

---

### Discussion Questions

Before moving to the chunking solution, consider:

1. **Why can't we just use a larger embedding model?** Even with larger context windows, there are performance and cost trade-offs. Plus, semantic search works better with focused chunks.

2. **What information might we lose when splitting a document?** Context that spans across chunk boundaries could be harder to retrieve.

3. **How would you decide where to split a document?** By paragraphs? By headings? By a fixed character count?

4. **What metadata should we preserve?** When you chunk a document, what information should each chunk retain about its source?

---

### What's Next?

You've successfully identified the problem: **large documents exceed embedding model token limits**. 

In the next part of this lab, you'll implement **document chunking** - a strategy to intelligently split the EmployeeHandbook.md into smaller, semantically meaningful pieces that can each be embedded and searched independently.

**Save your work** - you'll build on this code in the next section.

---
