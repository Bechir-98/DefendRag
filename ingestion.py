from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm


def load_all_pdfs(directory):
    documents=[]

    pdf_files=list(Path(directory).rglob("*.pdf"))

    for pdf_path in tqdm(pdf_files,desc="Loading PDFs"):
        reader=PdfReader(pdf_path)

        for page_number,page in enumerate(reader.pages,start=1):
            text=page.extract_text()

            if text and text.strip():
                documents.append({
                    "text":text,
                    "source":pdf_path.name,
                    "path":str(pdf_path),
                    "page":page_number
                })

    return documents


def build_chunks(docs,size=1000,overlap=150):
    chunks=[]

    for doc in docs:
        text=doc["text"]
        start=0

        while start<len(text):
            chunk_text=text[start:start+size]

            chunks.append({
                "chunk_id":len(chunks),
                "text":chunk_text,
                "source":doc["source"],
                "path":doc["path"],
                "page":doc["page"]
            })
            start+=size-overlap

    return chunks


if __name__=="__main__":
    docs=load_all_pdfs("data")
    print("Pages loaded:",len(docs))

    chunks=build_chunks(docs)
    print("Chunks:",len(chunks))
