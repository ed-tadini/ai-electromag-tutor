from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import VectorStore
from enhancer import Enhancer
from critic import Critic
from dotenv import load_dotenv
import time


load_dotenv()

class Pipeline:
    
    def __init__(self, vectorstore_path: str = "./chromadb"):
        self.llm = ChatOpenAI(model= 'gpt-4o', temperature = 0.4)
        self.system_prompt = """" You are an electromagnetism tutor that has a strong background in 
        first principles thinking. You will receive a student's question paired with:
        - an expert review of the question and possible learning gaps
        - context from private documents relevant to the question
        Your job is to provide to the student an answer based 70/100 on the context provided and
        30/100 based on your own knowledge, citing the context source and page number.
        """
        self.vector_store = VectorStore(persist_directory = vectorstore_path)
        self.vector_store.load_vectorstore()

    
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

        chuncks = self.refine_retrieval(enhancer_message['for_llm'], initial_chuncks, user_query)

        # Step 4: set up llm call

        expert_review = "EXPERT REVIEW\n" + "\n".join(enhancer_message['for_llm'])

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

response = rag.execute(question)

print('LLM response\n\n' + response)


