"""
RAG Fusion sample
"""
import argparse
import os
import sys
from operator import itemgetter

from langchain.load import dumps, loads
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter

# LLM model
LLM_MODEL_OPENAI = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# Retriever options
TOP_K = 5
MAX_DOCS_FOR_CONTEXT = 8
DOCUMENT_PDF = "Transformer Explainer (1).pdf"


# .env
os.environ['OPENAI_API_KEY'] = ''

my_template_prompt = """Please answer the [question] using only the following [information]. If there is no [information] available to answer the question, do not force an answer.

Information: {context}

Question: {question}
Final answer:"""

def load_and_split_document(pdf: str) -> list[Document]:
    """Load and split document

    Args:
        url (str): Document URL

    Returns:
        list[Document]: splitted documents
    """

    # Read the text documents from 'pdf'
    raw_documents = PyPDFLoader(pdf).load()

    # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)
    # Split the documents
    documents = text_splitter.split_documents(raw_documents)

    # for TEST
    print("Original document: ", len(documents), " docs")

    return documents

def create_retriever(search_type: str, kwargs: dict) -> BaseRetriever:
    """Create vector retriever

    Args:
        search_type (str): search type 
        kwargs (dict): kwargs

    Returns:
        BaseRetriever: Retriever
    """


    # load and split document
    documents = load_and_split_document(DOCUMENT_PDF)

    # chroma db
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(documents, embeddings)

    # retriever
    retriever = vectordb.as_retriever(
        search_type=search_type,
        search_kwargs=kwargs,
    )

    return retriever

def reciprocal_rank_fusion(results: list[list], k=60):
    """Rerank docs (Reciprocal Rank Fusion)

    Args:
        results (list[list]): retrieved documents
        k (int, optional): parameter k for RRF. Defaults to 60.

    Returns:
        ranked_results: list of documents reranked by RRF
    """

    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # for TEST (print reranked documentsand scores)
    print("Reranked documents: ", len(reranked_results))
    for doc in reranked_results:
        print('---')
        print('Docs: ', ' '.join(doc[0].page_content[:100].split()))
        print('RRF score: ', doc[1])

    # return only documents
    return [x[0] for x in reranked_results[:MAX_DOCS_FOR_CONTEXT]]

def query_generator(original_query: dict) -> list[str]:
    """Generate queries from original query

    Args:
        query (dict): original query

    Returns:
        list[str]: list of generated queries 
    """

    # original query
    query = original_query.get("query")

    # prompt for query generator
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
        ("user", "Generate multiple search queries related to:  {original_query}. When creating queries, please refine or add closely related contextual information in English, without significantly altering the original query's meaning"),
        ("user", "OUTPUT (3 queries):")
    ])

    # LLM model
    model = ChatOpenAI(
                temperature=0,
                model_name=LLM_MODEL_OPENAI
            )

    # query generator chain
    query_generator_chain = (
        prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # gererate queries
    queries = query_generator_chain.invoke({"original_query": query})

    # add original query
    queries.insert(0, "0. " + query)

    # for TEST
    print('Generated queries:\n', '\n'.join(queries))

    return queries

def rrf_retriever(query: str) -> list[Document]:
    """RRF retriever

    Args:
        query (str): Query string

    Returns:
        list[Document]: retrieved documents
    """

    # Retriever
    retriever = create_retriever(search_type="similarity", kwargs={"k": TOP_K})

    # RRF chain
    chain = (
        {"query": itemgetter("query")}
        | RunnableLambda(query_generator)
        | retriever.map()
        | reciprocal_rank_fusion
    )

    # invoke
    result = chain.invoke({"query": query})

    return result

def query(query: str, retriever: BaseRetriever):
    """
    Query with vectordb
    """

    # model
    model = ChatOpenAI(
        temperature=0,
        model_name=LLM_MODEL_OPENAI)

    # prompt
    prompt = PromptTemplate(
        template=my_template_prompt,
        input_variables=["context", "question"],
    )

    # Query chain
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question")
        }
        | RunnablePassthrough.assign(
            context=itemgetter("context")
        )
        | {
            "response": prompt | model | StrOutputParser(),
            "context": itemgetter("context"),
        }
    )

    # execute chain
    result = chain.invoke({"question": query})

    return result

def main():
    # OpenAI API KEY
    if os.environ.get("OPENAI_API_KEY") == "":
        print("`OPENAI_API_KEY` is not set", file=sys.stderr)
        sys.exit(1)

    # args (if you need them)
    # args = parser.parse_args()

    # Define the query
    query_text = 'what is ragchecker'

    # Create retriever
    retriever = RunnableLambda(rrf_retriever)

    # Call the query function
    result = query(query_text, retriever)

    # Print answer
    print('---\nAnswer:')
    print(result['response'])

if __name__ == '__main__':
    main()
