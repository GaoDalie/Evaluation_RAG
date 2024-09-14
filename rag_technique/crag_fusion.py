import io
import os
import operator
from typing import List, TypedDict, Sequence, Annotated
from langchain_core.messages import BaseMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import TokenTextSplitter
import streamlit as st
<<<<<<< HEAD
from langchain_community.tools.tavily_search import TavilySearchResults
=======
>>>>>>> origin/main

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5
MAX_DOCS_FOR_CONTEXT = 8
<<<<<<< HEAD
DOCUMENT_PDF = "Transformer Explainer (1).pdf"

os.environ["TAVILY_API_KEY"] = ""

os.environ["OPENAI_API_KEY"] = ""
=======
DOCUMENT_PDF = "2402.03367v2.pdf"


st.title("Multi-PDF ChatBot using LLAMA3 & Adaptive RAG")
user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')

with st.sidebar:
    uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
    process = st.button("Process")
>>>>>>> origin/main

class GraphState(TypedDict):
    llm_opus: ChatOpenAI # Claude 3 "Haiku" model
    emb_model: OpenAIEmbeddings # Embedding model
    question: str # Question text
    generate_querys: List[str] # Generated (additional) questions
    generate_query_num: int # Number of generated (additional) questions
    integration_question: str # Integrated question
    transform_question: str # Question transformed into a query for web search
    messages: Annotated[Sequence[BaseMessage], operator.add] # History of messages
    fusion_documents: List[List[Document]] # Documents retrieved from the generated questions
    documents: List[Document] # Documents ultimately passed to the LLM
    is_search: bool # Whether a web search is required




# Generate similar questions from the original question
def generate_query(state:GraphState) -> GraphState:
    print("\n--- __start__ ---")
    print("--- generate_query ---")
    llm = state["llm_opus"]
    question = state["question"]
    generate_query_num = state["generate_query_num"]
    system_prompt = "You are an assistant that generates multiple search queries based on a single input query."
    human_prompt = """When creating queries, output each query on a new line without significantly changing the original query's meaning.
    Input query: {question}
    {generate_query_num} output queries: 
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
    )
    questions_chain = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    generate_querys = questions_chain.invoke(
        {
            "question": question, 
            "generate_query_num": generate_query_num
        }
    )
    generate_querys.insert(0, "0. " + question)
    print("\nOriginal Question + Generated Questions==========================")
    for i, query in enumerate(generate_querys):
        print(f"\n{query}")
    print("\n===========================================================\n")
    
    return {"generate_querys": generate_querys}



# Print the updated state with generated queries
# print("Updated state with generated queries:")
# print(state)

def retrieve(state:GraphState) -> GraphState:
    print("--- retrieve ---")
    print(state)
<<<<<<< HEAD
    emb_model = state.get('emb_model')
    if emb_model is None:
         print("Error: 'emb_model' key not found in the state") 
    generate_querys = state["generate_querys"]
    raw_documents = PyPDFLoader(DOCUMENT_PDF).load()
    # Define chunking strategy
    print(raw_documents)
=======
    emb_model = state['emb_model']   
    generate_querys = state["generate_querys"]
    raw_documents = PyPDFLoader(DOCUMENT_PDF).load()
    # Define chunking strategy
>>>>>>> origin/main
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)
    # Split the documents
    documents = text_splitter.split_documents(raw_documents)

    print("Original document: ", len(documents), " docs")

    # chroma db
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(documents, embeddings)

    fusion_documents = []
    for question in generate_querys:
        docs = vectordb.similarity_search(question, k=3)
        fusion_documents.append(docs)
    return {"fusion_documents": fusion_documents}


# Calculate document scores and extract the top ones
def fusion(state):
    print("--- fusion ---")
    fusion_documents = state["fusion_documents"]
    k = 60
    documents = []
    fused_scores = {}
    for docs in fusion_documents:
        for rank, doc in enumerate(docs, start=1):
            if doc.page_content not in fused_scores:
                fused_scores[doc.page_content] = 0
                documents.append(doc)
            fused_scores[doc.page_content] += 1 / (rank + k)
    reranked_results = {doc_str: score for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:3]}
    print("\nTop 3 search scores ========================================")
    for i, score in enumerate(reranked_results.values(), start=1):
        print(f"\nDocument {i}: {score}")
    print("\n===========================================================\n")
    filterd_documents = []
    for doc in documents:
        if doc.page_content in reranked_results:
            filterd_documents.append(doc) 
    documents = filterd_documents
    return {"documents": documents}


def integration_query(state):
    print("--- integration_query ---")
    llm = state["llm_opus"]
    generate_querys = state["generate_querys"]
    system_prompt = """You are a question rewriter that consolidates multiple input questions into one question."""
    human_prompt = """Please output only the integrated question.
    Multiple questions: {query}
    Integrated question: """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    integration_chain = prompt | llm | StrOutputParser()
    questions = "\n".join(generate_querys)
    integration_query = integration_chain.invoke({"query": questions})
    print(f"\nIntegrated question: {integration_query}\n")
    return {"integration_question": integration_query}


def grade_documents(state):
    print("--- grade_documents ---")
    llm = state["llm_opus"]
    integration_question = state["integration_question"]
    documents = state["documents"]
    system_prompt = """You are an assistant that evaluates the relevance between searched documents and user questions.
    If the document contains keywords or semantic content related to the question, you evaluate it as relevant.
    Respond with "Yes" for relevance and "No" for no relevance."""
        
    human_prompt = """
        
    Document: {context} 
        
    Question: {query}
    Relevance ("Yes" or "No"): """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    filtered_docs = []
    is_search = False
    grade_chain = prompt | llm | StrOutputParser()
    print("\nEvaluation of relevance for each document =============================")
    for doc in documents:
        grade = grade_chain.invoke({"context":doc.page_content, "query": integration_question})
        print(f"\nRelevance: {grade}")
        if "Yes" in grade:
            filtered_docs.append(doc)
        else:
            is_search = True
    print("\n===========================================================\n")
    return {"documents": filtered_docs, "is_search": is_search}


def decide_to_generate(state):
    print("--- decide_to_generate ---")
    is_search = state['is_search']
    if is_search == True:
        return "transform_query"
    else:
        return "create_message"
    
def transform_query(state):
    print("--- transform_query ---")
    llm = state["llm_opus"]
    integration_question = state["integration_question"]
    system_prompt = """You are a rewriter that converts input questions into queries optimized for web search."""
    human_prompt = """Look at the question and infer the fundamental meaning/intent to output only the web search query.
    Question: {query}
    Web search query: """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    transform_chain = prompt | llm | StrOutputParser()
    transform_query = transform_chain.invoke({"query": integration_question})
    print(f"\nWeb search query: {transform_query}\n")
    state["transform_question"] = transform_query
    return {"transform_question": transform_query}

def web_search(state):
    print("--- web_search ---")
    transform_question = state["transform_question"]
    documents = state["documents"]
<<<<<<< HEAD
    retriever = TavilySearchResults()
=======
    retriever = SearchApiAPIWrapper()
>>>>>>> origin/main
    docs = retriever.run(transform_question)
    documents.extend(docs)
    
    return {"documents": documents}


def create_message(state):
    print("--- create_message ---")
    documents = state["documents"]
    question = state["question"]
    system_message = "You will always respond in English."
    human_message = """Refer to the context separated by '=' signs below to answer the question.


    {context}

    Question: {query}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", human_message),
        ]
    )
    partition = "\n" + "=" * 20 + "\n"
    valid_documents = [doc for doc in documents if hasattr(doc, 'page_content')]
    documents_context = partition.join([doc.page_content for doc in valid_documents])
    messages = prompt.format_messages(context=documents_context, query=question)

    return {"messages": messages}


def generate(state):
    print("--- generate ---")
    llm = state["llm_opus"]
    messages = state["messages"]
    response = llm.invoke(messages)
    print("--- end ---\n")
    
    return {"messages": [response]}


def get_compile_graph():
    graph = StateGraph(GraphState)
    graph.set_entry_point("generate_query")
    graph.add_node("generate_query", generate_query)
    graph.add_edge("generate_query", "retrieve")
    graph.add_node("retrieve", retrieve)
    graph.add_edge("retrieve", "fusion")
    graph.add_node("fusion", fusion)
    graph.add_edge("fusion", "integration_query")
    graph.add_node("integration_query", integration_query)
    graph.add_edge("integration_query", "grade_documents")
    graph.add_node("grade_documents", grade_documents)
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "create_message": "create_message"
        },
    )
    graph.add_node("transform_query", transform_query)
    graph.add_edge("transform_query", "web_search")
    graph.add_node("web_search", web_search)
    graph.add_edge("web_search", "create_message")
    graph.add_node("create_message", create_message)
    graph.add_edge("create_message", "generate")
    graph.add_node("generate", generate)
    graph.add_edge("generate", END)

    compile_graph = graph.compile()
    
    return compile_graph

<<<<<<< HEAD
# if process:

llm_opus = ChatOpenAI(model_name="gpt-4o")

emb_model =OpenAIEmbeddings(model="text-embedding-3-small")

compile_graph = get_compile_graph()
print(compile_graph)
# Ensure that the key 'GPT-4o' is added to the state dictionary
# Simplified state dictionary
state = {
    "llm_opus": llm_opus,  # Renamed the key to match the input structure
    "question": "What is  RAGChecker ", 
    "generate_query_num": 2
}

print("State dictionary before invoking compile_graph:", state)
# Invoke the graph with explicit parameters
output = compile_graph.invoke(
    input={
        "llm_opus": state["llm_opus"],
        "question": state["question"],
        "generate_query_num": state["generate_query_num"],
    }
)

print("output:")
print(output["messages"][-1].content)
=======
if process:

    llm_opus = ChatOpenAI(model_name="gpt-4o")

    emb_model =OpenAIEmbeddings(model="text-embedding-3-small")

    compile_graph = get_compile_graph()
    print(compile_graph)
    # Ensure that the key 'GPT-4o' is added to the state dictionary
    # Simplified state dictionary
    state = {
        "llm_opus": llm_opus,  # Renamed the key to match the input structure
        "question": "What is Rag Fusion", 
        "generate_query_num": 2
    }

    print("State dictionary before invoking compile_graph:", state)
    # Invoke the graph with explicit parameters
    output = compile_graph.invoke(
        input={
            "llm_opus": state["llm_opus"],
            "question": state["question"],
            "generate_query_num": state["generate_query_num"],
        }
    )
    
    st.write("output:")
    st.write(output["messages"][-1].content)
>>>>>>> origin/main
   

