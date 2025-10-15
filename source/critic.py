from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
imort re

load_dotenv()

### CRITIC BLUE PRINT ###
# Relevance ranking: best chuncks on top
# End goal 6 to 8 chuncks
# Flag chunck with cut concepts -> look up proximity search methods