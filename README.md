# Rick: an ai electromag tutor
Basic agentic RAG pipeline for physics electromagnetism.

## Architecture
Diagram of the high level architecture:
<div align="center">
<img src="readme/Rich_arch2.png" width="800" height="800">
</div>

### Critic Agent

#### What it does
Takes retrieved chunks and filters them down to the best 6-8 pieces using two AI experts:
1. **Ranking expert** - Sorts chunks by relevance to the question
2. **Relations expert** - Picks the best subset, removes redundancy

#### Returns
- `filtered_chunks` - The selected chunks
- `metadata` - Contains `confidence` (0.0 to 1.0), `reasoning`, and `num_selected`

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
