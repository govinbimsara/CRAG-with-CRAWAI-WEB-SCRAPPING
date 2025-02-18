import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import faiss
import pickle

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, CrawlerMonitor, DisplayMode

from langchain_groq import ChatGroq
from langchain.storage import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize LLM and Embeddings
llm = ChatGroq(model=os.getenv("LLM_MODEL", "mixtral-8x7b-32768"), api_key=os.getenv("GROQ_API_KEY"))
embeddings_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# FAISS Storage
# vectorstore = FAISS.load_local("faiss_index") if os.path.exists("faiss_index") else FAISS(embeddings_model)
'''
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
else:
    dimension = 384  # `all-MiniLM-L6-v2` outputs 384-dimensional embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index
    
    # Use an InMemoryStore to allow adding new documents
    docstore = InMemoryStore()
    
    # Initialize FAISS vector store with proper docstore
    vectorstore = FAISS(embedding_function=embeddings_model, index=index, docstore=docstore, index_to_docstore_id={})
'''

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end)
    return chunks


async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using Groq LLM."""
    system_prompt = """Extract a title and summary from the given content. 
    Return a JSON object in the format: {"title": "...", "summary": "..."}. 
    Do not include any other text or explanation."""

    # Invoke the LLM model
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
    ])
    # Ensure we extract the content properly
    if hasattr(response, "content"):
        response_text = response.content
    else:
        response_text = str(response)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}, response text: {response_text}")
        return {"title": "Error", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding using Hugging Face model."""
    return embeddings_model.embed_query(text)

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk."""
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)
    metadata = {
        "source": "genie_business_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    return ProcessedChunk(url, chunk_number, extracted['title'], extracted['summary'], chunk, metadata, embedding)

'''async def insert_chunk(chunk: ProcessedChunk):
    """Insert chunk into FAISS."""
    doc = Document(page_content=chunk.content, metadata={"title": chunk.title, "summary": chunk.summary, "url": chunk.url})
    vectorstore.add_documents([doc])
    vectorstore.save_local("faiss_index")
    print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}") '''

async def insert_chunk(chunk: ProcessedChunk):
    """Insert chunk into FAISS using add_texts."""
    global vectorstore  # Ensure we update the global vectorstore

    doc = Document(page_content=chunk.content, metadata={"title": chunk.title, "summary": chunk.summary, "url": chunk.url})

    # If FAISS index exists, load it; otherwise, create a new one
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
    else:
        # Create FAISS index on first insert
        vectorstore = FAISS.from_documents([doc], embeddings_model)
    
    # Add new document to the FAISS store
    vectorstore.add_documents([doc])
    
    vectorstore.save_local("faiss_index")
    print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")

async def process_and_store_document(url: str, markdown: str):
    """Process and store document chunks."""
    chunks = chunk_text(markdown)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    await asyncio.gather(*[insert_chunk(chunk) for chunk in processed_chunks])

async def crawl_batch():
    """Crawl multiple URLs in batch."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(memory_threshold_percent=70.0, check_interval=1.0, max_session_permit=10, monitor=CrawlerMonitor(display_mode=DisplayMode.DETAILED))
    urls = [
        "https://www.geniebusiness.lk/",
        "https://www.geniebusiness.lk/lending.php",
        "https://www.geniebusiness.lk/pricing.php",
        "https://www.geniebusiness.lk/developer.php",
        "https://www.geniebusiness.lk/downloads/Terms-and-Conditions-Genie-Business.pdf",
        "https://www.geniebusiness.lk/contact.php",
        "https://www.geniebusiness.lk/tap-to-pay.php",
        "https://www.geniebusiness.lk/payment-gateway.php",
        "https://www.geniebusiness.lk/multi-currency-pricing.php",
        "https://www.geniebusiness.lk/e-store.php",
        "https://www.geniebusiness.lk/payment-links.php",
        "https://www.geniebusiness.lk/qr-payments.php",
        "https://www.geniebusiness.lk/billing-services.php",
        "https://www.geniebusiness.lk/retail.php",
        "https://www.geniebusiness.lk/ecommerce-and-social-media-business.php",
        "https://www.geniebusiness.lk/travel-and-tourism.php",
        "https://www.geniebusiness.lk/home-based-business.php",
        "https://www.geniebusiness.lk/professional-services.php",
        "https://www.geniebusiness.lk/medium-and-large-corporates.php",
        "https://www.geniebusiness.lk/other-businesses.php"
    ]
    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls=urls, config=run_config, dispatcher=dispatcher)
        for result in results:
            if result.success:
                await process_and_store_document(result.url, result.markdown_v2.raw_markdown)
            else:
                print(f"Failed to crawl {result.url}: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(crawl_batch())
