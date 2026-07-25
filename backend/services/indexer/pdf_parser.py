from pathlib import Path
import pdfplumber
import json



def parse_pdf(pdf_path : Path):
    document={
        #pdf_path.name comes from Pdf_path which is a path obj
        #t has property .name
        "document_name":pdf_path.name,
        "pages":[]
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages,start=1):
            document["pages"].append(
                {
                    "page_number":page_number,
                    "content":page.extract_text() or ""
                }
            )
    return document

def parse_directory(pdf_directory:Path)->list[dict]:

    documents=[]
    
    #pdf_path is a path object
    for pdf_path in pdf_directory.glob("*.pdf"):
        documents.append(parse_pdf(pdf_path))
    
    return documents

if __name__ == "__main__":
    #ABSOLUTE PATH->Path(__file__).resolve(), resolve makes it absolute
    BASE_DIR = Path(__file__).resolve().parents[3]
    PDF_DIR = BASE_DIR/ "RAG_Documents"
    OUTPUT_DIR = Path(__file__).resolve().parent/"parsed_documents.json"
    documents = parse_directory(PDF_DIR)
    with OUTPUT_DIR.open("w",encoding="utf-8") as json_file:
        json.dump(documents,json_file,indent=4)

    print("JSON created successfully")
