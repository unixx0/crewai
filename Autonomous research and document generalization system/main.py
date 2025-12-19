from crewai import Agent, Task, Crew, LLM, Process
import os
from crewai_tools import DirectorySearchTool, TavilySearchTool, SerpApiGoogleSearchTool
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
from typing import List

os.environ['SERPAPI_API_KEY']= os.getenv('serp')
os.environ["TAVILY_API_KEY"] = os.getenv('tailvy')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

class structured(BaseModel):
    Answer: str= Field(description= "The answer of the given user Query")
    Sources: List[str]= Field(description= "The sources from where the agent retrieved the answer")



rag_tool= DirectorySearchTool(
    directory= '.\datasources',
    config= {
        'embedding_model': {

        'provider': 'huggingface',
        'config':{
            'url': 'https://api-interface.huggingface.co/sentence-transformers/all-MiniLM-L6-v2'
        }
    },
    'vectordb': {
        'provider': 'chromadb',
        'config': {}
    }
    }
)

tavily= TavilySearchTool()
serpapi= SerpApiGoogleSearchTool()

llm= LLM(
      model="huggingface/openai/gpt-oss-120b",
    
)

router_agent= Agent(
    role= "Routing agent",
    goal= '''Analyze the given user query: {query}
    1. Identify the type of query whether it is based on real time or on pdf's.
    2. Assign the query to the agent who have access to pdf and check analyze the output. If not required/sufficient output is received then assgin the query to agent having web access
    3. Analyze the final output and analyze whether it answers the user query. {query}''',
    backstory= 'You are good at knowing which agent to invoke on the basis of the requirement of user maintaining the correct flow and delegate task to respective agent',
    llm= llm,
    allow_delegation = True,
    verbose= True
    )

knowledge_agent= Agent(
    role= 'Pdf information Retriever',
    goal= '''
1. Analyze the given user query: {query},
2. Retrive the revelant information from the pdf on the basis of user query: {query}
3. If there is no any relevant information then you just say you dont know
''',
backstory= "You are provided at multiple pdf. You are the one who is good for retriving relevant information from that pdf whenever someone ask any questions.",
llm= llm,
tools= [rag_tool],

)

web_agent= Agent(
    role= 'Web Analyzer',
    goal= '''
1. Analyze the given user query: {query},
2. Retrive the relevant information from the web that answers the user query,
''',
backstory= 'You have a power to go through the web and search anything in the web',

tools= [tavily, serpapi],
llm= llm
)

task1= Task(
    description= 'Retirive the relevant informaation from pdf on the basis of user query: {query}',
    expected_output= 'Brief Answer to the given user query. If there is no relevant information then return dont know.',
     #guardrail= '''1. The output must be correct answer of the given user query,
    #2. The answer must be in professional and academic tone
    #3. If there agent cannot find the relevant answer then the output must be dont know.''',
    
)

task2= Task(
    description= 'Retirive the relevant informaation from web on the basis of user query: {query}',
    expected_output= 'Brief Answer to the given user query',
    #guardrail= '''1. The output must be correct answer of the given user query,
    #2. The answer must be in professional and academic tone
    #3. If there agent cannot find the relevant answer then the output must be dont know.''',

)

manager_task= Task(
    description= 'Analyze the answer from pdf and web. Synthesize them into a single answer and do comparision if needed',
    expected_output= '''An json containing Answer and Sources used to get answer''',
    output_pydantic= structured
)

crew= Crew(
    agents= [knowledge_agent, web_agent],
    tasks= [task1, task2, manager_task],
    verbose= True,
    manager_agent= router_agent,
    process= Process.hierarchical,
    tracing= True, 
    


)

user= '''What is the 2025 AI market outlook according to our internal reports, and how does this compare to the actual stock performance of C3.ai over the last 30 days?'''

result= crew.kickoff({'query': user})
print(result)
