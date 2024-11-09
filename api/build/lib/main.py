import dotenv
import os
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import (
    create_tool_calling_agent,
    Tool,
    AgentExecutor,
)
from langchain_core.prompts import ChatPromptTemplate
from tools import get_current_wait_time

dotenv.load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

chat_model = ChatGroq(
    model="llama-3.1-70b-versatile",
)

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="hospital_data",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_KEY"),    
)

reviews_retriever = qdrant.as_retriever(k=10)

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

tools = [
    Tool(
        name="Reviews",
        func=review_chain.invoke,
        description="""Useful when you need to answer questions
        about patient reviews or experiences at the hospital.
        Not useful for answering questions about specific visit
        details such as payer, billing, treatment, diagnosis,
        chief complaint, hospital, or physician information.
        Pass the entire question as input to the tool. For instance,
        if the question is "What do patients think about the triage system?",
        the input should be "What do patients think about the triage system?"
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """,
    ),
]

verbose_prompt = """When providing your final answer, include all relevant details 
                    from your observations."""
hospital_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", verbose_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

hospital_agent = create_tool_calling_agent(
    llm=chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_agent_executor = AgentExecutor(
    agent=hospital_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)

print(hospital_agent_executor.invoke({"input": "What is the current wait time at hospital C?"} ))
print(hospital_agent_executor.invoke({"input": "What have patients said about their comfort at the hospital?"} ))
