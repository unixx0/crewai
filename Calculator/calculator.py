from crewai import Crew, Agent, Task,LLM
from crewai.tools import tool, BaseTool
import os
from dotenv import load_dotenv
load_dotenv()



@tool
def add(a: float, b: float) -> float:
    ''' This tool is used to add the numbers '''
    
    return a + b

@tool
def subtraction(a: float, b: float) -> float:
    ''' This tool is used to subtract the numbers '''
    
    return a - b

@tool
def multiplication(a: float, b: float) -> float:
    ''' This tool is used to multiply (*) the numbers '''
    
    return a * b


@tool
def division(a: float, b: float) -> float:
    ''' This tool is used divide  numbers where a is dividend and b is divisior'''
    
    
    return a / b


@tool
def modulus(a: float, b: float) -> float:
    ''' This tool is used perform modulus operation(%) between  numbers where a is dividend and b is divisior and returns remainder of a/b'''
    
    return a % b

llm= LLM(
    model= 'huggingface/openai/gpt-oss-120b',
    temperature= 0.5,
    api_key= os.getenv('hugging')
)

llm2= LLM(
    model= 'huggingface/meta-llama/Llama-3.1-8B-Instruct',
    api_key= os.getenv('hugging')
)

calculator_agent = Agent(
    role="Calculator",
    goal="Solve math queries using tools",
    backstory='''You are a calculator ie simply a machine. You cannot do mathematical calculations of your own. But you are provided tools  that helps you solving mathematical problems and YOU MUST USE TOOLS PROVIDED TO YOU to solve the task. Without using tools you are always incorrect. You are purely dependent on tools.'
    The only rule you know is the sequence of execution of the operations ie BODMAS from left to right where
    B= Bracket,
    O= Of,
    D= Division,
    M= Multiplication,
    A= Addition and
    S= Subtraction.
    This will help you in solving the mathematical problems where there are multiple operations USING TOOLS''',
    llm=llm2,
    function_calling_llm=llm,
    allow_delegation= False,  
    tools=[add, subtraction, multiplication, division, modulus]
)

from pydantic import BaseModel, Field
class structured_output(BaseModel):
    Question: str= Field(description= "The given user question")
    Answer: float= Field(description= "The calculated answer")


task= Task(
    description= "This task is related for performing mathematical calculations on the basis of given user query: {query}. The agent MUST USE TOOLS to solve the query",
    expected_output= "The answer of given user mathematical problem: {query}",
    agent= calculator_agent,
    output_pydantic= structured_output,
    

)


calculator= Crew(
    agents= [calculator_agent],
    tasks= [task],
    planning= True,
    planning_llm= llm,
    verbose= True
)

query= input("Enter the required Question: ")

result= calculator.kickoff({'query': query})

print(f'The answer is {result.pydantic.Answer}')
                           