from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import VectorStore
from enhancer import Enhancer
from critic import Critic
from dotenv import load_dotenv


load_dotenv()

class Pipeline:
    
    def __init__(self, vectorstore_path: str = "./chromadb"):
        self.llm = ChatOpenAI(model= 'gpt-4o', temperature = 0.4)
        self.vector_store = VectorStore(persist_directory = vectorstore_path)
        self.vector_store.load_vectorstore()

    def retrieve_chuncks(self, enhancer_context: dict, user_query: str = '', 
                    initial_num: int = 20, max_attempts: int = 2, 
                    confidence_threshold: float = 0.7):
        
        if user_query == '':
            raise ValueError('no user query recieved by retrival method')

        # Step 1: raw semantic retrieval
        initial_chunks = self.vector_store.search(user_query, k=initial_num)
        
        # Step 2: critic review and feedback loop
        for attempt in range(max_attempts):
            critic = Critic(
                enhancer_message=enhancer_context,
                chunks=initial_chunks
            )
            filtered_chunks, critic_metadata = critic.execute()
            
            # Accept if confidence sufficient or max attempts reached
            if critic_metadata['confidence'] >= confidence_threshold or attempt == max_attempts - 1:
                break
            
            # Low confidence: get more chunks
            more_chunks = self.vector_store.search(user_query, k=10)
            initial_chunks.extend(more_chunks)

        return filtered_chunks