from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import VectorStore
from enhancer import Enhancer
from critic import Critic
from dotenv import load_dotenv
import time


load_dotenv()

class Pipeline:
    
    def __init__(self, vectorstore_path: str = "./chromadb"):
        self.llm = ChatOpenAI(model= 'gpt-4o', temperature = 0.4)
        self.vector_store = VectorStore(persist_directory = vectorstore_path)
        self.vector_store.load_vectorstore()

    
    def refine_retrieval(self, enhancer_context: dict, initial_chuncks: list[tuple[any, float]], 
                    user_query: str = '', initial_num: int = 20, max_attempts: int = 2, 
                    confidence_threshold: float = 0.7):
        
        if user_query == '':
            raise ValueError('no user query recieved by refine retrival method')

        for attempt in range(max_attempts):
            critic = Critic(
                enhancer_message=enhancer_context,
                chunks=initial_chuncks
            )
            filtered_chunks, critic_metadata = critic.execute()
            
            # Accept if confidence sufficient or max attempts reached
            if critic_metadata['confidence'] >= confidence_threshold or attempt == max_attempts - 1:
                break
            
            # Low confidence: get more chunks
            more_chunks = self.vector_store.search(user_query, k=10)
            initial_chuncks.extend(more_chunks)

        return filtered_chunks
    
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

