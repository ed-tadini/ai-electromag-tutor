from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
from dataclasses import field
import re

load_dotenv()

### CRITIC BLUE PRINT ###
# Relevance ranking: best chuncks on top
# End goal 6 to 8 chuncks
# Flag chunck with cut concepts -> look up proximity search methods

class Critic():
    enhacer_message = str
    loop_number = int
    experts = field(default_factory = dict)
    chuncks = field(default_factory = list)

    def __init__(self, ref_message: str, chuncks: list[str]):
        self.enhacer_message = ref_message
        self.loop_number = 0
        self.chuncks = chuncks

        self.experts = {
            'ranking expert' : ChatPromptTemplate.from_messages([
                ('system', """You are a renowed physics professor in electromagnetism know for your fundamental
                understanding of math and physics. You are given different chuncks from some documents, and 
                YOUR GOAL is to rank them from best to worst in relation to what the student needs and is trying 
                to understand. 
                
                Your input will be:
                - REQUIREMENTS : an abstract from another professor reporting the student's question
                paired with his review
                - CHUNCKS: list of chuncks to rank.
                
                FORMAT YOUR RESPONSE AS:
                - [RANKED NUMBER] = [CHUNCK] for approximaly the total number of chuncks divided by 2."""),
                ('user', """
                -REQUIREMENTS = {ref_message}
                -CHUNCKS = {chuncks}
                """)
            ]),
            'relations expert' : ChatPromptTemplate.from_messages([
                ('system', """
                
                """),
                ('user', """
                
                """)
            ])
        }


