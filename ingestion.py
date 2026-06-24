from langchain_core.documents.base import Document


import os
import ssl
from typing import Any, Dict, List
import asyncio

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilyCrawl,TavilyExtract,TavilyMap, tavily_crawl, tavily_map
from logger import log_info, log_success, log_error, log_warning, log_header, Colors

load_dotenv()

# Configure SSL context to user certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CONTEXT"] = str(ssl_context)
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

#Initialise embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",show_progress_bar="false",chunk_size=50, retry_min_seconds=10)
vectorStore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """ Process Documents in batches asynchronously"""
    log_header("Vector storage Phase")
    log_info(f"Indexing {len(documents)} documents in batches of {batch_size}",Colors.DARKCYAN)

    batches = [
        documents[i:i+batch_size] for i in range(0, len(documents), batch_size)
    ]


    async def add_batch(batch: List[Document], batch_number: int):
        """Add a batch of documents to the vector store"""
        try:
            await vectorStore.add_documents(batch)
            log_success(f"Batch {batch_number} of {len(batch)} documents added to the vector store")
        except Exception as e:
            log_error(f"Error adding batch {batch_number} of {len(batch)} documents: {e}")
            return False
        return True

    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks,return_exceptions=True)


    #Count successful batches
    successful = sum([1 for result in results if result is True])

    if successful == len(batches):
        log_success(f"All {len(batches)} batches were successfully added to the vector store")
    else:
        log_error(f"Only {successful} out of {len(batches)} batches were successfully added to the vector store")
        return False
    return True
    



async def main():
    """Main function to run the orchestrate the entire process"""
    log_header("Starting the document ingestion process")

    log_info("TavilyCrawl: Starting web crawl document from :https://docs.langchain.com/oss/python/langchain/overview ")

    #Crawl the document
    res = tavily_crawl.invoke(
        {
            "url": "https://docs.langchain.com/oss/python/langchain/overview",
            "max_depth": 3,
            "extract_depth":"advanced",
        }
    )

    all_docs = {}

    print(type(res))
    if(type(res) == dict):
        for doc in res["results"]:
            if doc["raw_content"] is not None:
                all_docs = [Document(page_content=doc["raw_content"], metadata={"source": doc["url"]})]
            else:
                log_warning(f"TavilyCrawl: Raw content is None for URL: {doc['url']}")
                continue
        # all_docs = [Document(page_content=doc["raw_content"], metadata={"source": doc["url"]}) for doc in res["results"]]
        log_success(f"TavilyCrawl: Successfully crawled the document with {len(all_docs)} pages")
    else:
        all_docs = res
        log_error(f"TavilyCrawl: Error crawling the document: {res}")

 
    #split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"TavilyCrawl: Successfully split the document into {len(splitted_docs)} chunks")

    #index the documents
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("Document ingestion process completed successfully")
    log_success(f"Document ingestion process completed successfully with {len(splitted_docs)} chunks indexed")
    log_info(f"Total documents indexed: {len(all_docs)}",Colors.DARKCYAN)
    log_info('chunks indexed successfully',Colors.GREEN)





if __name__ == "__main__":
    asyncio.run(main())