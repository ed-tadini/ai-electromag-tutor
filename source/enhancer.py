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
                KEYWORDS: [key terms for retrieval, comma-separated]
                
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
                CONNECTIONS: [1 sentence about links to other physics areas, if relevant]
                SEARCH_TERMS: [optimized terms for database retrieval, comma-separated]"""),
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
            ])
        }
        
        self.parallel_chain = RunnableParallel(**{
            name: prompt | self.llm 
            for name, prompt in self.experts.items()
        })
    
    def enhance(self, user_query):
        """
        Run all experts and return outputs for retrieval and LLM context.
        
        Return: dictionary -> keys = 'query', 'for_retrival', 'for_llm'
        """
        if not user_query:
            raise ValueError("No query provided")
        
        outputs = self.parallel_chain.invoke({'query': user_query}) #API endpoint 1

        semantic = outputs['semantic_expert'].content
        physics = outputs['physics_expert'].content
        prerequisites = outputs['prerequisite_expert'].content
        
        #for retrival
        search_terms = self._extract_search_terms(semantic, physics)

        #====Monitoring===
        print(f"\n{'='*60}")
        print("ENHANCER DEBUG")
        print(f"{'='*60}")
        print(f"Original query: {user_query}")
        print(f"\nExtracted search terms ({len(search_terms)}): {search_terms}")
        print(f"\nSemantic analysis:\n{semantic[:200]}...")
        print(f"\nPhysics concepts:\n{physics[:200]}...")
        print(f"\nPrerequisites:\n{prerequisites[:200]}...")
        #==================
        
        return {
            'query': user_query,
            
            'for_retrieval': search_terms,
            
            'for_llm': {
                'semantic': semantic,
                'physics': physics,
                'prerequisites': prerequisites
            }
        }
    
    def _extract_search_terms(self, semantic, physics):

        terms = []
        

        keywords_match = re.search(r'KEYWORDS:(.+?)(?:\n|$)', semantic)
        if keywords_match:
            terms.extend([k.strip() for k in keywords_match.group(1).split(',')])

        core_match = re.search(r'CORE:(.+?)(?:\n|$)', physics)
        if core_match:
            terms.extend([k.strip() for k in core_match.group(1).split(',')])
        
        search_match = re.search(r'SEARCH_TERMS:(.+?)(?:\n|$)', physics)
        if search_match:
            terms.extend([k.strip() for k in search_match.group(1).split(',')])
        
        return list(set(terms))


#======= test =======

enhancer = Enhancer()
result = enhancer.enhance("can you explain to me Gauss law and how I can use it to find electric fields?")

print("=== SEARCH WITH THESE ===")
print(result['for_retrieval'])



print("\n=== GIVE THIS CONTEXT TO LLM ===")
print(result['for_llm']['semantic'])
print(result['for_llm']['physics'])
print(result['for_llm']['prerequisites'])



