from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import List, Tuple, Any
import json

load_dotenv()

class Critic:
    def __init__(self, enhancer_message: str, chunks: List[Tuple[Any, float]]):
        self.enhancer_message = enhancer_message
        self.chunks = chunks
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.ranking_prompt = ChatPromptTemplate.from_messages([
            ('system', """You are a renowned physics professor specializing in electromagnetism.
            TASK: Rank document chunks by relevance to the student's question.
            
            OUTPUT FORMAT (JSON):
            {{
                "rankings": [
                    {{"id": 0, "rank": 1, "score": 0.95, "reason": "Brief explanation"}},
                    {{"id": 1, "rank": 2, "score": 0.87, "reason": "Brief explanation"}}
                ]
            }}

            Rank from most relevant (rank=1) to least relevant."""),
            ('user', """
            REQUIREMENTS: {enhancer_message}
            CHUNKS: {formatted_chunks}
            """)
        ])
        
        self.filter_prompt = ChatPromptTemplate.from_messages([
            ('system', """You are a logic expert. Filter ranked chunks to create a coherent picture.

                GOAL: Select the BEST chunks that form logical connections.
                - Prefer 6-8 chunks, but prioritize quality
                - Eliminate redundancy while maintaining completeness
                - Keep diverse perspectives

                OUTPUT FORMAT (JSON):
                {{
                    "selected_ids": [0, 3, 5, 7, 9, 11],
                    "reasoning": "Brief explanation of selection logic",
                    "confidence": 0.85
                }}

                confidence = 1.0 means perfect coverage, 0.0 means critical gaps exist."""),
            ('user', """RANKED CHUNKS: {ranked_chunks}""")
        ])
    
    def _format_chunks(self) -> str:

        formatted = []
        for i, (doc, score) in enumerate(self.chunks):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            formatted.append(f"[ID: {i}] [Search Score: {score:.3f}]\n{content}\n")
        return "\n".join(formatted)
    
    def rank_chunks(self) -> dict:

        #=====Monitoring=====
        print(f"\n{'='*60}")
        print("CRITIC - RANKING STAGE")
        print(f"{'='*60}")
        print(f"Total chunks to rank: {len(self.chunks)}")
        print(f"Enhancer context length: {len(self.enhancer_message)} chars")
        #====================

        chain = self.ranking_prompt | self.llm | JsonOutputParser()
        result = chain.invoke({
            'enhancer_message': self.enhancer_message,
            'formatted_chunks': self._format_chunks()
        }) #API endpoint 2

        #=====Monitoring=====
        print(f"Ranked {len(result['rankings'])} chunks")
        print(f"Top 3 ranks: {result['rankings'][:3]}")
        #====================

        return result
    
    def filter_chunks(self, ranked_result: dict) -> dict:
        
        #=====Monitoring=====
        print(f"\n{'='*60}")
        print("CRITIC - FILTERING STAGE")
        print(f"{'='*60}")
        #====================

        ranked_str = json.dumps(ranked_result['rankings'], indent=2)
        
        chain = self.filter_prompt | self.llm | JsonOutputParser()
        result = chain.invoke({
            'ranked_chunks': ranked_str
        }) #API endpoint 3

        #=====Monitoring=====
        print(f"Selected {len(result['selected_ids'])} chunks: {result['selected_ids']}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')[:150]}...")
        #====================
        
        return result
    
    
    def execute(self) -> tuple[List[Tuple[any, float]], dict]:
        """Main execution pipeline -> returns chunks and metadata"""
        # Step 1: Rank all chunks
        ranked = self.rank_chunks()
        
        # Step 2: Filter to best chunks
        filtered = self.filter_chunks(ranked)
        
        # Step 3: Return filtered chunks + metadata
        selected_ids = filtered['selected_ids']
        assert all(i < len(self.chunks) for i in selected_ids), f"Invalid IDs: {selected_ids}, only have {len(self.chunks)} chunks"
        final_chunks = [self.chunks[i] for i in selected_ids]
        
        metadata = {
            'confidence': filtered.get('confidence', 0.8),
            'reasoning': filtered.get('reasoning', ''),
            'num_selected': len(selected_ids)
        }
        
        return final_chunks, metadata

