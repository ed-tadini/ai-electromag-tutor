from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
import re

load_dotenv()

class Enhancer:  
    def __init__(self):
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
        
        self.experts = {
            'semantic_expert': ChatPromptTemplate.from_messages([
                ("system", """You are an experienced teacher evaluating what a student needs.
                
                Analyze their question and provide a brief assessment describing:
                - What type of explanations would help them most
                - Why you think they need this particular approach
                - Any learning gaps or misconceptions you detect
                
                Consider these categories as guidance: theory, applications, exercises, derivation, 
                examples, intuition, prerequisites, advanced topics, conceptual understanding, 
                problem-solving strategies.
                
                Format your response as:
                PRIMARY_NEED: [main type of content needed]
                ASSESSMENT: [2-3 sentences of pedagogical insight]
                
                Be insightful and pedagogical - read between the lines."""),
                ("user", "{query}")
            ]),
            
            'physics_expert': ChatPromptTemplate.from_messages([
                ("system", """You are a physics professor specializing in classical electromagnetism.
                
                Analyze the query and identify relevant physics concepts at two levels:
                
                CORE CONCEPTS: Physics concepts directly related to answering this question.
                List specific electromagnetic concepts (e.g., "electric field", "Gauss's law", 
                "magnetic flux", "Faraday's induction").
                
                BACKGROUND CONCEPTS: Foundational or connected concepts that provide broader context.
                These help students see the bigger picture and make connections.
                
                If strong overlaps exist with other physics fields (mechanics, thermodynamics, 
                optics, quantum), briefly mention them as "bridges to other fields."
                
                Format your response as:
                CORE: concept1, concept2, concept3
                BACKGROUND: concept1, concept2, concept3
                CONNECTIONS: [1 sentence about links to other physics areas, if relevant]"""),
                ("user", "{query}")
            ]),
            
            'prerequisite_expert': ChatPromptTemplate.from_messages([
                ("system", """You are an applied math expert at identifying prerequisite knowledge.
                
                For this electromagnetism question, identify:
                - Essential prerequisites the student MUST know both in math 
                    (specifically calculus or linear algebra) and in physics
                - Likely gaps if they're asking this question
                
                Format:
                PREREQUISITES: concept1, concept2, concept3
                LIKELY_GAPS: [1-2 sentences about what they might be missing]
                """),
                ("user", "{query}")
            ]),
            
            'query_reformulation_expert': ChatPromptTemplate.from_messages([
                ("system", """You are a search optimization expert for physics textbook retrieval.
                
                Your task: Transform the student's question into 2-3 enhanced search queries that will 
                retrieve the most relevant textbook content from a vector database.
                
                Guidelines:
                - Expand abbreviations and implicit concepts
                - Add technical terms that would appear in textbook explanations
                - Make queries specific enough to avoid noise but broad enough to capture relevant content
                - Include both the "what" (concept) and "how" (application/derivation) if relevant
                - Each query should be a complete, natural sentence or phrase (not keywords)
                - Think about how textbooks actually explain these concepts
                
                Format your response as:
                QUERY1: [first enhanced query]
                QUERY2: [second enhanced query]
                QUERY3: [third enhanced query, optional]
                
                Example:
                Student asks: "how does Gauss law work?"
                QUERY1: Gauss law for electric fields and its mathematical formulation
                QUERY2: applying Gauss law to calculate electric field using symmetry
                QUERY3: integral form of Gauss law and closed surface charge enclosed
                """),
                ("user", "{query}")
            ])
        }
        
        self.parallel_chain = RunnableParallel(**{
            name: prompt | self.llm 
            for name, prompt in self.experts.items()
        })
    
    def enhance(self, user_query: str) -> dict:
        if not user_query:
            raise ValueError("No query provided")
        
        # run all experts in parallel
        outputs = self.parallel_chain.invoke({'query': user_query}) #API call 1
        
        semantic = outputs['semantic_expert'].content
        physics = outputs['physics_expert'].content
        prerequisites = outputs['prerequisite_expert'].content
        reformulated = outputs['query_reformulation_expert'].content
        
        # extract enhanced queries for retrieval
        enhanced_queries = self._extract_enhanced_queries(reformulated)
        
        return {
            'query': user_query,
            'for_retrieval': enhanced_queries,
            'for_llm': {
                'semantic': semantic,
                'physics': physics,
                'prerequisites': prerequisites
            }
        }
    
    def _extract_enhanced_queries(self, reformulated_text: str) -> list[str]:

        queries = []
        
        for i in range(1, 4):
            pattern = rf'QUERY{i}:(.+?)(?:QUERY{i+1}:|$)'
            match = re.search(pattern, reformulated_text, re.DOTALL)
            if match:
                query = match.group(1).strip()
                query = ' '.join(query.split())
                if query:
                    queries.append(query)

        if not queries:
            for line in reformulated_text.split('\n'):
                if line.strip().startswith('QUERY'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        query = parts[1].strip()
                        if query:
                            queries.append(query)
        
        return queries if queries else []


# ======= TESTING =======
if __name__ == "__main__":

    enhancer = Enhancer()
    query = "If you had to pinpoint only 1 concept or equation or law to best describe electrostatics, what would that be?"
    result = enhancer.enhance(query)


    print(f"\n{'='*80}")
    print(f"ORIGINAL QUERY: {query}")
    print('='*80)
    
    
    print("\n=== ENHANCED QUERIES FOR RETRIEVAL ===")
    for i, q in enumerate(result['for_retrieval'], 1):
        print(f"{i}. {q}")
    
    print("\n=== PEDAGOGICAL CONTEXT FOR LLM ===")
    print("\nSemantic Expert:")
    print(result['for_llm']['semantic'])
    print("\nPhysics Expert:")
    print(result['for_llm']['physics'])
    print("\nPrerequisite Expert:")
    print(result['for_llm']['prerequisites'])



