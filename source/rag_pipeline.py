from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from enhancer import Enhancer
from critic import Critic
from dotenv import load_dotenv
import time


load_dotenv()

class Pipeline:
    
    def __init__(self, vectorstore_path: str = "./chromadb"):
        self.llm = ChatOpenAI(model= 'gpt-4o', temperature = 0.4)
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
            - At the end of the message list the prerequisites mentioned by the pedagocial expert
            that we assumed the student knew

            Remember: You're teaching, not just answering. Help them build intuition.
        """
        embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embeddings
        )

    
    def refine_retrieval(self, enhancer_context: dict, initial_chuncks: list[tuple[any, float]], 
                    user_query: str = '', max_attempts: int = 3, confidence_threshold: float = 0.7):
        
        if user_query == '':
            raise ValueError('no user query recieved by refine retrival method')

        CRITIC_APPROVAL = False
        
        for attempt in range(max_attempts):
            critic = Critic(
                enhancer_message=enhancer_context,
                chunks=initial_chuncks
            )
            filtered_chunks, critic_metadata = critic.execute()
            
            # Accept if confidence sufficient or max attempts reached
            if critic_metadata['confidence'] >= confidence_threshold or attempt == max_attempts - 1:
                CRITIC_APPROVAL = True
                break
            
            # Low confidence: get more chunks
            more_chunks = self.vector_store.search(user_query, k=10)
            initial_chuncks.extend(more_chunks)
        
        if CRITIC_APPROVAL:
            return filtered_chunks
        else:
            raise ValueError('critic could not find relevant chuncks')
        
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

        # Step 1: enhance the user query

        enhancer = Enhancer()
        enhancer_message = enhancer.enhance(question)

        # Step 2: first retrieval

        CHAR_THRESHOLD = 50
        
        if len(question) < CHAR_THRESHOLD:
            user_query = ' '.join(enhancer_message['for_retrieval']) + question.lower()
        else:
            user_query = question.lower()

        INITIAL_CHUNCK_NUM = 20

        initial_chuncks = self.vector_store.similarity_search_with_relevance_scores(
            user_query,
            k = INITIAL_CHUNCK_NUM
        )

        # Step 3: critic selection and loop

        expert_review = "EXPERT REVIEW\n" + "\n\n".join(enhancer_message['for_llm'].values())

        chuncks = self.refine_retrieval(expert_review, initial_chuncks, user_query)

        # Step 4: set up llm call

        context = self.format_context(chuncks)

        message = ChatPromptTemplate.from_messages([
            ('system', f"{self.system_prompt}{expert_review}{context}"),
            ('user', f'{question}')
        ])

        # Step 5: final llm call

        pipeline_t = time.time()

        response = self.llm.invoke(message) #API endpoint 4

        llm_t = time.time()

        return response, pipeline_t-start_time, llm_t - pipeline_t


#======= test =======

question = 'can you explain to me Gauss Law and how I can use it to derive other equation when solving problems'

rag = Pipeline()

response, pipeline_time, llm_time = rag.execute(question)

print('LLM response\n\n' + response + '\nPipeline time: ' + pipeline_time +'\nLLM time: ' + llm_time)


