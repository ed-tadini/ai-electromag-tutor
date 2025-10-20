# Rick: an ai electromag tutor
Basic agentic RAG pipeline for physics electromagnetism.

## Architecture
Diagram of the high level architecture:
<div align="center">
<img src="readme/Rich_arch2.png" width="800" height="800">
</div>

### Enhancer Agent

#### What it does
Analyzes the user's question with three parallel AI experts to understand what they need:
1. **Semantic expert** - Identifies explanation type needed and learning gaps
2. **Physics expert** - Extracts core concepts and background knowledge
3. **Prerequisite expert** - Identifies required prerequisites and likely gaps

#### Returns
Dictionary with three keys:
- `query`: `str` - Original user question
- `for_retrieval`: `List[str]` - Optimized search terms for vector DB
- `for_llm`: `dict` - Context for personalized responses:
  - `semantic`: Pedagogical assessment
  - `physics`: Core concepts and connections
  - `prerequisites`: Required knowledge and gaps

#### Basic usage
```python
enhancer = Enhancer()
result = enhancer.enhance("can you explain Gauss law?")

# Use for retrieval
chunks = vectorstore.search(result['for_retrieval'], k=20)

# Pass context to LLM for personalized teaching
llm_context = result['for_llm']
```

#### Why use it
- Converts vague questions into precise search terms
- Gives LLM insight into what explanation style helps most
- All three experts run in parallel for speed

### Critic Agent

#### What it does
Takes retrieved chunks and filters them down to the best 6-8 pieces using two AI experts:
1. **Ranking expert** - Sorts chunks by relevance to the question
2. **Relations expert** - Picks the best subset, removes redundancy

#### Returns
- `filtered_chunks`: `List[Tuple[Document, float]]` - The selected chunks with their original similarity scores
- `metadata`: `dict` - Quality assessment containing:
  - `confidence`: `float` (0.0 to 1.0) - How well the chunks cover the question
  - `reasoning`: `str` - Why these chunks were selected
  - `num_selected`: `int` - Number of chunks selected

#### Basic usage
```python
critic = Critic(enhancer_message=enhanced_query, chunks=retrieved_chunks)
filtered_chunks, metadata = critic.execute()
```

#### Adding a retrieval loop
Use the `confidence` score to decide if you need more chunks:
```python
chunks = retriever.retrieve(enhanced_query, k=20)

for attempt in range(2):  # Try max 2 times
    critic = Critic(enhanced_query, chunks)
    filtered_chunks, metadata = critic.execute()
    
    # Good enough? Stop here
    if metadata['confidence'] >= 0.7 or attempt == 1:
        break
    
    # Low confidence? Get more chunks and try again
    more_chunks = retriever.retrieve(enhanced_query, k=10)
    chunks.extend(more_chunks)
```

#### When to use the loop
- Use it if answers often feel incomplete
- Skip it if retrieval already gets good chunks (saves cost)
- Start without it, add later if needed
