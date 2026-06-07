## Lab 4: Implementing Chunking Strategies with LangChain

In Lab 3, you discovered that the EmployeeHandbook.md is too large to embed as a single document. Now you'll solve this problem by implementing **document chunking** - splitting large documents into smaller, meaningful pieces that can each be embedded and searched independently.

LangChain provides built-in text chunking capabilities through the **langchain_text_splitters** module. In this lab, you'll explore different chunking strategies and understand when to use each one.

### Understanding Chunking Strategies

Different chunking strategies serve different purposes:

1. **Fixed-Size Chunking** - Splits text into chunks of a specific character or token count. Simple but may break in the middle of sentences or concepts.

2. **Paragraph-Based Chunking** - Splits on paragraph boundaries, preserving complete thoughts. Works well for well-formatted documents.

3. **Sentence-Based Chunking** - Splits on sentence boundaries, grouping sentences together up to a maximum size. Balances granularity with coherence.

4. **Token-Based Chunking** - Uses actual token counts (as the AI model sees them) rather than character counts, providing precise control over embedding limits.

Each strategy has trade-offs between semantic coherence, chunk size consistency, and implementation complexity.

---

### Step 1: Install Text Chunking Dependencies

LangChain's text chunking functionality is available through the **langchain-text-splitters** package.

**Prompt 1: Add Text Chunking Package**
```
I need to add text chunking capabilities to my Python project.
Add the langchain-text-splitters package to requirements.txt.
Show me the import statements I'll need for text splitting in LangChain.
```

**What to Look For:** The langchain-text-splitters package should be added to requirements.txt. You'll import various splitters like CharacterTextSplitter, RecursiveCharacterTextSplitter, and MarkdownHeaderTextSplitter.

**Documentation Reference:**
- [LangChain Text Splitters](https://python.langchain.com/v0.2/docs/how_to/#text-splitters)
- [Text Splitters API Reference](https://reference.langchain.com/python/langchain_text_splitters/)
- [Document Transformers](https://python.langchain.com/v0.2/docs/integrations/document_transformers/)

---

### Step 2: Create a Chunking Helper Function

Before implementing different strategies, create a generic function to handle chunking and storing.

**Prompt 2: Create Chunked Document Loading Function**
```
Create a function called load_document_with_chunks that:
- Takes parameters: vector_store, file_path, and a list of LangChain Document objects (chunks)
- Loops through each chunk with its index
- For each chunk:
  - Updates the chunk's metadata to include:
    - 'fileName': The file name plus " (Chunk X/Total)"
    - 'createdAt': Current timestamp using datetime.now().isoformat()
    - 'chunkIndex': The chunk number
  - Adds the chunk to the vector store using vector_store.add_documents()
- Prints progress for each chunk processed
- Returns the total number of chunks stored

Include error handling using try-except.
Import datetime from the datetime module.
```

**What to Look For:** This function will handle the common logic of processing chunks, regardless of which chunking strategy you use.

---

### Step 3: Implement Fixed-Size Character Chunking

The simplest strategy: split text every N characters.

**Prompt 3: Add Fixed-Size Chunking**
```
Create a function that:
- Reads the EmployeeHandbook.md file using open() and read()
- Uses CharacterTextSplitter from langchain_text_splitters with:
  - chunk_size=1000
  - chunk_overlap=0
  - separator=" " (split on spaces to avoid breaking words)
- Calls create_documents([text]) to generate Document objects
- Passes the chunks to load_document_with_chunks
- Prints statistics: number of chunks created, average chunk size
- Update the code to call `load_with_fixed_size_chunking` instead of `load_document` so the EmployeeHandbook is split before being added to the vector store.

Import CharacterTextSplitter from langchain_text_splitters.
```

**What to Look For:** The document should be split into multiple chunks of roughly equal size.

**Documentation:**
- [CharacterTextSplitter API](https://reference.langchain.com/python/langchain_text_splitters/)
- [Python File I/O](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)

**Test Point**: Run the application using `python app.py`. You should see the EmployeeHandbook successfully split and loaded.

**Expected Output:**
```
📄 Loading EmployeeHandbook.md with fixed-size chunking...
Processing chunk 1/15...
Processing chunk 2/15...
...
✅ Successfully loaded 15 chunks from EmployeeHandbook.md
Average chunk size: 1,247 characters
```

---

### Step 4: Implement Paragraph-Based Chunking

A more semantic approach: split on paragraph boundaries while respecting size limits.

**Prompt 4: Add Paragraph Chunking**
```
Create a function to chunk by paragraphs:
- Read the EmployeeHandbook.md file
- Use RecursiveCharacterTextSplitter from langchain_text_splitters with:
  - chunk_size=1500
  - chunk_overlap=0
  - separators=["\n\n", "\n", " ", ""] (splits on paragraphs first, then newlines, then spaces)
- This splitter tries to keep paragraphs together while respecting the size limit
- Call create_documents([text]) to generate chunks
- Pass the chunks to load_document_with_chunks
- Compare the number of chunks and their sizes to fixed-size chunking
- Update the code to call the new method so the EmployeeHandbook is splitin this new method.

Print a comparison showing:
- Total chunks created
- Size of smallest and largest chunks
- How many chunks start with a newline (indicating paragraph preservation)

Import RecursiveCharacterTextSplitter from langchain_text_splitters.
```

**What to Look For:** Chunks should align with document structure, making them more semantically coherent.

**Documentation:**
- [RecursiveCharacterTextSplitter API](https://reference.langchain.com/python/langchain_text_splitters/)
- [Text Splitters How-to Guide](https://python.langchain.com/v0.2/docs/how_to/#text-splitters)

**Do not test!**: Do not run at this point, these are just exploration functions.

---

### Step 5: Implement Markdown-Aware Chunking

For markdown documents, respect the document structure (headings, sections).

**Prompt 5: Add Markdown Structure Chunking**
```
Create a function to chunk markdown by structure:
- Read the EmployeeHandbook.md file
- Use MarkdownHeaderTextSplitter from langchain_text_splitters with:
  - headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
- This splits the document on markdown headers, preserving structure
- Then apply RecursiveCharacterTextSplitter with:
  - chunk_size=5000
  - chunk_overlap=200
- The overlap helps preserve context across chunk boundaries
- Pass the chunks to load_document_with_chunks
- Update the code to call the new method so the EmployeeHandbook is splitin this new method.

Import MarkdownHeaderTextSplitter from langchain_text_splitters.
```

**What to Look For:** Chunks should preserve markdown structure, keeping headings with their content. Check the metadata for header information.

**Documentation:**
- [MarkdownHeaderTextSplitter API](https://reference.langchain.com/python/langchain_text_splitters/)
- [Markdown Splitting Guide](https://python.langchain.com/v0.2/docs/how_to/#text-splitters)

**Why Overlap Matters:** When you split "...benefits include health insurance. \n\n Health insurance covers..." at the paragraph break, overlap ensures both chunks contain "health insurance" context, improving search results.

**Do not test!**: Do not run at this point, these are just exploration functions.

---

### Step 6: Adding an Agent to Complete the RAG Pattern!

Now that you have implemented chunking and vector search, let's complete the **Retrieval-Augmented Generation (RAG)** pattern by creating an AI agent that can search your documents and answer questions using the retrieved context.

Instead of just returning raw chunks from the vector database, an agent can:
1. Search the vector database for relevant chunks
2. Use those chunks as context when answering the user's question
3. Generate a natural language response based on the retrieved information

You'll implement this step-by-step by prompting GitHub Copilot to help you create the necessary components.

---

**Sub-Step 6.1: Create a Search Tool for the Agent**

The first step is to create a LangChain tool that the AI agent can call to search your document collection.

**Prompt 6.1: Create Document Search Tool**
```
Create a function called create_search_tool that:
- Takes vector_store as a parameter
- Defines an inner function called search_documents that:
  - Accepts a query string parameter
  - Uses vector_store.similarity_search_with_score(query, k=3) to get top 3 results
  - Formats the results as a string with content and scores
  - Returns: "Result 1 (Score: X.XXXX): [content]\n\nResult 2..."
- Converts search_documents to a LangChain Tool using the @tool decorator from langchain_core.tools
- Add a docstring to search_documents describing: "Searches the company document repository for relevant information based on the given query. Use this to find information about company policies, benefits, and procedures."
- Returns the tool

Import tool from langchain_core.tools.

Note: The @tool decorator converts a Python function into a tool that agents can use.
The docstring is important - it tells the agent when and how to use the tool.
```

**What to Look For:** 
- The tool should use LangChain's @tool decorator pattern
- The docstring provides clear guidance to the agent about when to use this tool
- The search function should return formatted results that provide context to the LLM

**Documentation:**
- [LangChain Tools](https://python.langchain.com/docs/how_to/#tools)
- [Creating Custom Tools](https://python.langchain.com/docs/how_to/custom_tools/)
- [Tool Decorator](https://reference.langchain.com/core/tools/)

---

**Sub-Step 6.2: Add Chat Model**

Next, add a chat model that will power the conversational agent.

**Prompt 6.2: Add Chat Model**
```
Add code to create a chat model:
- Import ChatOpenAI from langchain_openai
- Create a ChatOpenAI instance with:
  - model="gpt-4o"
  - temperature=0 (for consistent, factual responses)
  - Use the same base_url and api_key from the embeddings configuration
- Add this after creating the vector store and before the interactive loop

Note: This uses the GitHub Models API endpoint, just like the embeddings.
```

**What to Look For:**
- The chat model should use the GitHub Models API (same endpoint as embeddings)
- Model "gpt-4o" provides good balance of quality and speed
- Temperature 0 ensures consistent, factual responses

**Documentation:**
- [ChatOpenAI](https://reference.langchain.com/langchain_openai/)
- [LangChain Chat Models](https://python.langchain.com/docs/integrations/chat/)

---

**Sub-Step 6.3: Create the Agent with the Search Tool**

Now create an agent that can use your search tool to answer questions.

**Prompt 6.3: Create ReAct Agent**
```
Create an agent using LangChain's ReAct pattern:
- Import create_react_agent from langchain.agents
- Import AgentExecutor from langchain.agents
- Import ChatPromptTemplate and MessagesPlaceholder from langchain_core.prompts
- Create the search tool by calling create_search_tool(vector_store)
- Create a prompt template with:
  - System message: "You are a helpful assistant that answers questions about company policies, benefits, and procedures. Use the search_documents tool to find relevant information before answering. Always cite which document chunks you used in your answer."
  - MessagesPlaceholder for chat_history
  - User message: {input}
  - MessagesPlaceholder for agent_scratchpad
- Create the agent using create_react_agent(llm=chat_model, tools=[search_tool], prompt=prompt)
- Create an AgentExecutor with:
  - agent=agent
  - tools=[search_tool]
  - verbose=True (to see the agent's reasoning)
  - handle_parsing_errors=True

Note: The ReAct pattern allows the agent to Reason and Act iteratively.
The agent_scratchpad is where the agent tracks its thought process.
```

**What to Look For:**
- The agent should be created with the search tool
- The prompt should guide the agent to use the search tool
- verbose=True will show you when the agent calls the search tool

**Documentation:**
- [LangChain Agents](https://python.langchain.com/docs/how_to/#agents)
- [ReAct Agent](https://python.langchain.com/docs/how_to/agent_executor/)
- [Agent Prompts](https://python.langchain.com/docs/how_to/agent_prompt/)

---

**Sub-Step 6.4: Replace the Interactive Search Loop with Agent Chat**

Finally, replace the simple vector search CLI with an agent-powered chat interface.

**Prompt 6.4: Create Agent Chat Loop**
```
Replace the existing semantic search while loop with a new chat interface that:

1. Creates an empty list called chat_history to track the conversation
2. Prints a welcome message explaining the agent's capabilities
3. In a loop:
   - Prompt the user: "You: "
   - Read user input
   - Exit on "quit" or "exit"
   - Call agent_executor.invoke({
       "input": user_input,
       "chat_history": chat_history
     })
   - Extract the response from result["output"]
   - Print the assistant's response with a clear prefix like "Agent: "
   - Add the user message and agent response to chat_history as HumanMessage and AIMessage
   - Import HumanMessage and AIMessage from langchain_core.messages

The agent should automatically call the search_documents tool when needed to answer questions.

Note: Because verbose=True, you'll see the agent's reasoning process including when it calls the search tool.
```

**What to Look For:**
- The agent should automatically invoke the search_documents tool when it needs information
- Conversation history should be maintained across multiple turns using chat_history
- The agent's responses should reference the document chunks it retrieved
- verbose=True will show the agent's "thought process"

**Documentation:**
- [Agent Executor](https://python.langchain.com/docs/how_to/agent_executor/)
- [Message Types](https://python.langchain.com/docs/concepts/messages/)
- [Chat History](https://python.langchain.com/docs/how_to/chatbots_memory/)

---

**Sub-Step 6.5: Test the Agent**

Run your application and test the agent-powered RAG system.

**Example Conversation:**
```
You: What health insurance benefits does the company offer?
[Agent reasoning shows it's calling search_documents("health insurance benefits")]
Agent: Based on the company documentation, we offer comprehensive health insurance...

You: How do I enroll?
[Agent reasoning shows it's calling search_documents("health insurance enrollment")]
Agent: To enroll in health insurance benefits, you need to...

You: What about dental coverage?
[Agent reasoning shows it's calling search_documents("dental coverage")]
Agent: According to the benefits documentation...
```

**What to Look For:**
- The agent should automatically search documents when it needs information
- Responses should be natural and conversational, not just returning raw chunks
- The agent should maintain context across the conversation
- The agent should cite which document chunks it used
- With verbose=True, you can see the agent's reasoning in the terminal

---

**Discussion: RAG vs. Simple Search**

After completing Step 6, compare the agent-powered RAG approach to the simple vector search from earlier steps:

**Simple Vector Search (Steps 1-5):**
- Returns raw document chunks
- User must read through chunks to find the answer
- No synthesis or summarization
- Limited to exact matches

**Agent-Powered RAG (Step 6):**
- Automatically searches when needed
- Synthesizes information from multiple chunks
- Provides natural language answers
- Can handle follow-up questions with conversation context
- Cites sources for transparency

This is the complete RAG pattern used in production AI applications!

---

### Discussion Questions

After completing Lab 4, consider:

1. **Chunk Size Trade-offs**: What happens if chunks are too small (100 characters)? Too large (5000 characters)? What's the "sweet spot"?

2. **Context Boundaries**: Did you notice any queries where the answer was split awkwardly across chunks? How could overlap help?

3. **Strategy Selection**: When would you choose fixed-size vs. markdown-aware chunking? What about for different document types (code, legal documents, chat logs)?

4. **Metadata Loss**: What information about document structure is lost when chunking? How could you preserve it?

5. **Search Quality**: Did chunking improve or worsen search results compared to searching full documents? Why?

---

### Extension Challenges

If you want to explore further:

**Challenge 1: Dynamic Chunk Sizing**
```
Ask Copilot: "Implement adaptive chunk sizing that creates smaller chunks for dense technical content and larger chunks for narrative text"
```

**Challenge 2: Hierarchical Chunking**
```
Ask Copilot: "Create a two-level chunking system: large chunks for context and smaller chunks for precision, storing both in the vector database with parent-child relationships"
```

**Challenge 3: Chunk Quality Scoring**
```
Ask Copilot: "Add a quality score to each chunk based on: completeness (starts/ends with sentence boundaries), coherence (single topic), and optimal size"
```

**Challenge 4: Smart Overlap**
```
Ask Copilot: "Implement intelligent overlap that only overlaps when chunks split mid-concept, detected by checking if text ends with sentence punctuation"
```

**Challenge 5: Cross-Chunk Context**
```
Ask Copilot: "When returning search results, if a chunk is ranked highly, also return the chunks immediately before and after it for better context"
```

---

### Key Takeaways

By completing Lab 4, you've learned:

✅ **Chunking is Essential** - Large documents must be split to work with embedding models

✅ **Multiple Strategies Exist** - Fixed-size, paragraph-based, and structure-aware chunking each have their place

✅ **Trade-offs Matter** - Chunk size, overlap, and boundary detection all affect search quality

✅ **Context Preservation** - Good chunking strategies preserve semantic meaning and document structure

✅ **Testing is Critical** - Always test chunking strategies with real queries to ensure quality

You now understand one of the core engineering challenges in RAG systems. In production applications, chunking strategy can make or break the user experience!

---

## What's Next?

You've now mastered the fundamentals of RAG:
- ✅ **Embeddings** - Converting text to vectors
- ✅ **Vector Storage** - Storing and indexing embeddings
- ✅ **Semantic Search** - Finding similar content
- ✅ **Chunking** - Handling large documents

The next step would be integrating RAG with an LLM to answer questions using retrieved context - but that's a topic for another lab!

**Great work completing this RAG engineering fundamentals unit!** 🎉