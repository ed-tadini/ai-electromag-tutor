from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import List, Tuple
import json

load_dotenv()

class Critic:
    def __init__(self, enhancer_message: str, chunks: List[Tuple[any, float]]):
        self.enhancer_message = enhancer_message
        self.chunks = chunks
        self.loop_number = 0
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Ranking expert
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
                        ('user', """REQUIREMENTS: {enhancer_message}

            CHUNKS:
            {formatted_chunks}""")
        ])
        
        # Relations expert
        self.filter_prompt = ChatPromptTemplate.from_messages([
            ('system', """You are a logic expert. Filter ranked chunks to create a coherent picture.

            GOAL: Select 6-8 most relevant chunks that form logical connections.
            - Eliminate redundancy
            - Keep diverse perspectives
            - Maintain logical flow

            OUTPUT FORMAT (JSON):
            {{
                "selected_ids": [0, 3, 5, 7, 9, 11],
                "reasoning": "Brief explanation of selection logic"
            }}"""),
                        ('user', """RANKED CHUNKS:
            {ranked_chunks}""")
                    ])
    
    def _format_chunks(self) -> str:
        """Format chunks for ranking"""
        formatted = []
        for i, (doc, score) in enumerate(self.chunks):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            formatted.append(f"[ID: {i}] [Search Score: {score:.3f}]\n{content}\n")
        return "\n".join(formatted)
    
    def rank_chunks(self) -> dict:
        """Rank chunks by relevance"""
        chain = self.ranking_prompt | self.llm | JsonOutputParser()
        result = chain.invoke({
            'enhancer_message': self.enhancer_message,
            'formatted_chunks': self._format_chunks()
        })
        return result
    
    def filter_chunks(self, ranked_result: dict) -> dict:
        """Filter chunks for logical connections"""
        # Format ranked chunks for filtering
        ranked_str = json.dumps(ranked_result['rankings'], indent=2)
        
        chain = self.filter_prompt | self.llm | JsonOutputParser()
        result = chain.invoke({
            'ranked_chunks': ranked_str
        })
        return result
    
    def execute(self) -> List[Tuple[any, float]]:
        """Main execution pipeline"""
        # Step 1: Rank all chunks
        ranked = self.rank_chunks()
        
        # Step 2: Filter to 6-8 best chunks
        filtered = self.filter_chunks(ranked)
        
        # Step 3: Return filtered chunks in ranked order
        selected_ids = filtered['selected_ids']
        final_chunks = [self.chunks[i] for i in selected_ids]
        
        self.loop_number += 1
        return final_chunks


