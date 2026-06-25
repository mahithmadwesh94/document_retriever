import os
from dotenv import load_dotenv
from typing import Dict, Any
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage,ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorStore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)
                                  
model = init_chat_model(model="gpt-5.5",temperature=0)
retriever = vectorStore.as_retriever(search_kwargs={"k": 5})
@tool(response_format="content_and_artifact")
def retrieve_context(query:str)->Dict[str, Any]: 
    """Retrieve Context for the query given by user from the vector store"""
    retrieved_docs = retriever.invoke(query)
    serialized_docs = "\n\n".join(
        f"source={doc.metadata.get('source')}\n\nContent:{doc.page_content}" for doc in retrieved_docs
    )
    return serialized_docs, retrieved_docs


def run_llm(query:str):
    """
    Run the RAG pipeline with user query to give answer based on retrieved documentation.

    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )


    agent = create_agent(model=model, tools=[retrieve_context], system_prompt=system_prompt)

    messages = [{"role":"user","content":query}]

    response = agent.invoke({"messages": messages})

    print(response)
    content = response["messages"][-1].content

    # Loop through messages and get the sources of the retrieved documents
    context_docs = []

    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            artifact = message.artifact
            if isinstance(artifact, list):
                context_docs.extend(artifact)
            else:
                context_docs.append(artifact)

    print("Length of context_docs:", len(context_docs))
    return {"answer": content, "context": context_docs}


    




if __name__ == "__main__":
    # Example usage
    query = "What is agent in LangChain?"
    result = run_llm(query)
    print("Answer:", result["answer"])
    print("="*100)
    print("Context:", result["context"])