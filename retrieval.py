import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage,ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings




load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorStore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)

chat_model = init_chat_model(model="gpt-5.5",temperature=0)
retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    """Retrieve context from the vector store"""
    retrieved_docs = retriever.invoke(query)

    serialized_docs = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent:{doc.page_content}" for doc in retrieved_docs
    )
    return serialized_docs, retrieved_docs


def run_llm(query:str):
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

    #create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(
        tools=[retrieve_context],
        model=chat_model,
        system_prompt=system_prompt
    )

    messages = [{"role":"user","content":query}]

    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content


    context_docs = []

    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            context_docs.append(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == "__main__":
    query = "What is the purpose of the LangChain library?"
    result = run_llm(query)
    print(result["answer"])
    print(result["context"])


    