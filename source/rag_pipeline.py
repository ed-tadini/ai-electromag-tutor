from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks import get_openai_callback
from dataclasses import dataclass
from dotenv import load_dotenv
from .enhancer import Enhancer
from .critic import Critic
import time


load_dotenv()

@dataclass
class PipelineConfig:
    initial_chunk_count: int = 20
    max_refinement_attempts: int = 2
    confidence_threshold: float = 0.7
    vector_db_path: str = "/home/ed-dev/Projects/ai-electromag-tutor/source/chromadb/chroma_db"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.4

class Pipeline:
    
    def __init__(self):
        self.config = PipelineConfig()
        self.llm = ChatOpenAI(model= self.config.llm_model, temperature= self.config.llm_temperature)
        self.enhancer = Enhancer()
        self.system_prompt = """You are an expert electromagnetism tutor with deep expertise in first principles thinking.
            You will receive:
            1. A student's question
            2. An expert pedagogical review identifying their learning needs
            3. Relevant context from electromagnetism textbooks and materials

            Your task:
            - Answer the student's question primarily using the provided context
            - Supplement with your own knowledge only when necessary to clarify or connect concepts
            - Always cite sources using the format: [Source: filename, Page: X]
            - Adapt your explanation style based on the pedagogical review
            - If the student shows misconceptions, address them gently but clearly
            - Use mathematical derivations when appropriate, but explain each step
            - For LaTeX math, use $$ for display equations and $ for inline equations
            - At the end of the message list the prerequisites mentioned by the pedagocial expert
            that we assumed the student knew

            Remember: You're teaching, not just answering. Help them build intuition.
        """
        embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = Chroma(
            persist_directory=self.config.vector_db_path,
            embedding_function=embeddings
        )

    def first_retrieval(self, queries: list[str], k: int = 7) -> list[tuple]:

        seen = {}
        
        #=====Monitoring=====
        print(f"\n{'='*60}")
        print("FIRST RETREIVAL")
        print(f"{'='*60}")
        print(f"\n{'\n'.join(queries)}")
        #====================

        for query in queries:
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            for doc, score in results:
                key = doc.page_content
                if key not in seen or score > seen[key][1]:
                    seen[key] = (doc, score)
        
        return sorted(seen.values(), key=lambda x: x[1], reverse=True)

    
    def refine_retrieval(self, enhancer_context: str, initial_chuncks: list[tuple[any, float]], 
                         queries: list[str], user_query: str = ''):
        
        if user_query == '':
            raise ValueError('no user query recieved by refine retrival method')
        if len(initial_chuncks) == 0:
            raise ValueError('no chuncks delivered to critic')

        #=====Monitoring=====
        print(f"\n{'='*60}")
        print("REFINEMENT LOOP")
        print(f"{'='*60}")
        #====================

        for attempt in range(self.config.max_refinement_attempts):
            critic = Critic(
                enhancer_message=enhancer_context,
                chunks=initial_chuncks
            )
            filtered_chunks, critic_metadata = critic.execute()

            if critic_metadata['confidence'] >= self.config.confidence_threshold:
                break
            elif attempt == self.config.max_refinement_attempts - 1:
                print('Critic run out of feedback attempts')
                break
            
            more_chunks = self.first_retrieval(queries, k = 5)
            initial_chuncks.extend(more_chunks)
            #=====Monitoring=====
            print(f"  Low confidence - fetching 10 more chunks")
            print(f"  Total chunks now: {len(initial_chuncks)}")
            #====================
        
        return filtered_chunks
        
    def format_context(self, chuncks: list[tuple[any, float]]) -> str:

        formatted : str = """CONTEXT:\n\n"""

        for doc, value in chuncks:
            format_doc: str = f"""SOURCE = {doc.metadata['source']} at page {doc.metadata['page']} \n
            CONTENT = {doc.page_content}\n\n
            """
            formatted += format_doc

        return formatted

    
    def execute(self, question: str = ''):
        
        if question == '':
            raise ValueError('no question received by the pipeline')
        
        start_time = time.time()

        #=====Monitoring=====
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION START")
        print(f"{'='*60}")
        #====================

        # Step 1: enhance the user query

        enhancer_message = self.enhancer.enhance(question)

        # Step 2: first retrieval

        initial_chuncks = self.first_retrieval(enhancer_message['for_retrieval'])

        #=====Monitoring=====
        print(f"\nInitial retrieval: {len(initial_chuncks)} chunks")
        #====================

        # Step 3: critic selection and loop

        expert_review = "PEDAGOGICAL REVIEW\n" + "\n\n".join(enhancer_message['for_llm'].values())

        chuncks = self.refine_retrieval(
            expert_review, 
            initial_chuncks, 
            enhancer_message['for_retrieval'],
            question
        )

        # Step 4: set up llm call

        context = self.format_context(chuncks)

        #=====Monitoring=====
        print(f"\nFinal context: {len(chuncks)} chunks selected")
        print(f"Context length: {len(context)} chars")
        print(f"\n{'='*60}")
        print("CALLING LLM")
        print(f"{'='*60}")
        #====================

        message = ChatPromptTemplate.from_messages([
            ('system', f"{self.system_prompt}{expert_review}{context}"),
            ('user', f'{question}')
        ])

        # Step 5: final llm call

        pipeline_t = time.time()

        with get_openai_callback() as cb:
            chain = message | self.llm
            response = chain.invoke({}) #API endpoint 4
            print(f"\nLLM Stats:")
            print(f"  Tokens: {cb.total_tokens}")
            print(f"  Cost: ${cb.total_cost:.4f}")

        llm_t = time.time()

        #=====Monitoring=====
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"pipeline time: {pipeline_t-start_time}")
        print(f"llm time: {llm_t - pipeline_t}\n")
        #====================

        return response


#======= test =======

question = "If you had to pinpoint only 1 concept or equation or law to best describe electrostatics, what would that be?"
rag = Pipeline()

response = rag.execute(question)

with open('response.md', 'w') as f:
    f.write(response.content)
print(f'Response saved to response.md')


