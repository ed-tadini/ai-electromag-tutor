import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def text_extractor(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                yield page_num, text

def process_pdf_streaming(pdf_path, chunk_size=512, chunk_overlap=128): #bigger chunck better for physics
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    for page_num, page_text in text_extractor(pdf_path):
        chunks = text_splitter.split_text(page_text)
        for chunk in chunks:
            yield Document(
                page_content=chunk,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "page": page_num
                }
            )

def process_and_store(pdf_folder, vector_db_path, batch_size=25): #batch_size = 25 -> 8GB of RAM taken

    embeddings = HuggingFaceEmbeddings(
        model_name='all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'batch_size': 16}
    )
    
    vector_db = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    batch_documents = []
    total_chunks = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        #chunks from PDF
        for doc in process_pdf_streaming(pdf_path):
            batch_documents.append(doc)
            total_chunks += 1
            
            #process in batches to control memory
            if len(batch_documents) >= batch_size:
                vector_db.add_documents(batch_documents)
                print(f"Stored batch of {len(batch_documents)} chunks (Total: {total_chunks})")
                batch_documents = []
    
    #remaining chunks
    if batch_documents:
        vector_db.add_documents(batch_documents)
        print(f"Stored final batch of {len(batch_documents)} chunks")
    
    
    print(f"\nTotal chunks processed: {total_chunks}")
    print(f"Vector database saved to: {vector_db_path}")


if __name__ == "__main__":
    process_and_store(
        pdf_folder="/home/ed-dev/Projects/ai-electromag-tutor/docs/",
        vector_db_path="chromadb/chroma_db",
        batch_size=12 
    )