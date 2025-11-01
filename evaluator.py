from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from source.rag_pipeline import Pipeline
import re

load_dotenv()

class Judge:
    def __init__(self):
        self.llm = ChatOpenAI(model= 'gpt-5', temperature = 0)
        self.prompt = ChatPromptTemplate.from_messages([
            ('system', """You are an expert evaluator specializing in physics education and pedagogical quality assessment. 
             Your task is to compare two answers to the same electromagnetism question and determine which provides superior educational value.
            
            YOUR ROLE
            You will receive:
            - Student Question: The original query about electromagnetism
            - Answer A (RAG): Response from the RAG-enhanced tutoring system
            - Answer B (Baseline): Response from a plain LLM without retrieval augmentation
            Your job is to evaluate Answer A against Answer B using the criteria below and assign a score.
             
            EVALUATION CRITERIAS
            1. First-Principles Reasoning 
            (Weight: 40%)
            What to look for:
            - Does the answer build understanding from fundamental physical principles?
            - Are core laws and equations derived or explained from their foundational assumptions?
            - Does it trace concepts back to Maxwell's equations, Coulomb's law, or other fundamental postulates?
            - Are physical quantities defined rigorously (what IS an electric field, not just what it does)?
            Red flags:
            - Jumping directly to formulas without explaining their physical origin
            - Presenting results as facts without derivation or justification
            - Missing the "why" behind mathematical relationships
            2. Conceptual Comprehensiveness 
            (Weight: 35%)
            What to look for:
            - Breadth of connections: Does it link the concept to related electromagnetic phenomena?
            - Cross-topic integration: Are connections made to mechanics, thermodynamics, or other physics domains when relevant?
            - Multiple perspectives: Does it offer both mathematical and intuitive/physical interpretations?
            - Prerequisite awareness: Does it acknowledge or briefly explain necessary background concepts?
            - Holistic understanding: Does it show how this concept fits into the larger framework of electromagnetism?
            Red flags:
            - Isolated explanations with no broader context
            - Single-perspective answers (only math OR only intuition, not both)
            - Missing obvious connections to related concepts
            3. Clarity and Pedagogical Quality 
            (Weight: 25%)
            What to look for:
            - Logical flow: Does the explanation build ideas progressively?
            - Precision: Are technical terms used correctly and defined when first introduced?
            - Accessibility: Can a student at the appropriate level understand this?
            - Examples and analogies: Are concrete examples or helpful analogies provided?
            - Misconception addressing: Does it preemptively clarify common confusions?
            Red flags:
            - Circular definitions or unclear explanations
            - Assuming too much or too little prior knowledge
            - Overly abstract without grounding examples
            - Confusing organization or logical jumps
             
            SCORING SYSTEM
            Assign one of these scores:
            - +1.0 (RAG answer is substantially better)
                Demonstrates clear superiority in at least 2 of the 3 criteria
                Shows significant depth in first-principles reasoning that the baseline lacks
                Provides rich conceptual connections that enhance understanding
                The pedagogical improvement is unmistakable
            - +0.5 (RAG answer is somewhat better)
                Shows improvement in 1-2 criteria but the difference is modest
                Provides some additional depth or connections but not transformative
                Both answers are adequate, but RAG adds noticeable value
                A knowledgeable student would appreciate the RAG answer more
            - 0.0 (Answers are equivalent or RAG is worse)
                Both answers are roughly equivalent in quality
                RAG answer doesn't add meaningful pedagogical value
                RAG answer may be longer but not substantively better
                RAG answer introduces errors, irrelevant information, or reduces clarity
                The baseline LLM answer is actually clearer or more helpful
            OUTPUT FORMAT
            Provide your evaluation in this structure:
             SCORING: +1.0 or +0.5 or 0.0
             REASONING: briefly why you chose the score
             """),
        ('user', """Student question: {query}
                    Answer A (RAG): {rag_answer}
                    Answer B: {plain_answer}
         """)
        ])

    def judge_answers(self, question: str, rag_answer: str, llm_answer: str) -> tuple[float, str]:

        chain = self.prompt | self.llm
        judge_answer = chain.invoke({
            'query' : question,
            'rag_answer' : rag_answer,
            'plain_answer' : llm_answer,
        }, query = question) #API call 

        response_text = judge_answer.content

        score_match = re.search(r'SCORING:\s*([+-]?\d+\.?\d*)', response_text)
        score = float(score_match.group(1)) if score_match else 0.0
        
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n\n|\Z)', response_text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return score, reasoning
        


def generate_replies(question: str) -> list[str]:
    
    comparison_llm = ChatOpenAI(model = 'gpt-4o')
    my_tutor = Pipeline()

    plain_answer = comparison_llm.invoke(question)
    my_answer = my_tutor.execute(question)

    return [my_answer, plain_answer]


def main():

    #10 samples for now
    test_questions_set: list[str] = ["Can you explain to me Gauss Law and how I can use it to solve problems?",
                                 "Can you explain to me what are the relations betwen Coloub forces, electrical field equation, electrical potential and charge distribution?",
                                 "If you had to pinpoint only 1 concept or equation or law to best describe electrostatics, what would that be?",
                                 "How can I calculate an electrical field of many (continuos) charges?",
                                 "How can I calculate the total work done by an ensemble of discrete charge?",
                                 "Can you help me truly get what Gauss law is, what it wants to describe, and its practical use?",
                                 "Can you tell me the main conditions or information I need to retain on ideal conductors when solving practical problems?",
                                 "Can you briefly go over on the derivations to get to Poisson's equation and why do we need to know this?",
                                 "Can you help me derive and understand the elctrical field of dipole?",
                                 "Can you explain to me how do we rewrite Ohm's law in relation to Maxwell's equations and why do we need this change?"] 
    

    scores: list[float] = []
    reasonings: list[str] = []
    judge = Judge()

    for i, test_question in enumerate(test_questions_set):

        answers = generate_replies(test_question)

        score, reasoning = judge.judge_answers(test_question, answers[0], answers[1])

        scores.append(score)
        reasonings.append(reasoning)
        print("\nJUDGEMENT DONE\n")

    mean_score = sum(scores) / len(scores)
    documentation = [str(item) for pair in zip(scores, reasonings) for item in pair]

    with open('evaluation_report.md', 'w') as f:
        f.write(f"MEAN SCORE = {mean_score}\n\n\n")
        f.write("\n".join(documentation))

    print("Report saved in evaluation_report.md")


    

if __name__ == '__main__':
    main()


