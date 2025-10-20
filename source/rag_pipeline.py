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

    def retrieve_chuncks(self, user_query: str = '', initial_num: int = 20):
        if user_query == '':
            raise ValueError('no user query recieved by retrival method')

        initial_chunks = self.vector_store.search(user_query, k=initial_num)
        ###all the part from first retrieval to critic, including the loop implementation###