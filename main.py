from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import io
from dotenv import load_dotenv
from fastapi import FastAPI , UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


app = FastAPI()


origins = ["https://chatwithmepdf.onrender.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just try to provide the answer from yourself if you can\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def get_conversational(user_question):

    prompt_template = """
    you are a expert in giving title to chat based on filename passed as question if you dont get any title just return chat with first two words of file name \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    chain = model.invoke("genrate a chat title in 3 words only based on filename - "+user_question)
    return chain.content



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


class Query(BaseModel):
    query: str
    uuid: str | None = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    content = await file.read()

    # Convert bytes to a memory file-like object
    pdf_file = io.BytesIO(content)

    raw_text = get_pdf_text([pdf_file])
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return {"status":raw_text}

@app.post("/chat/") 
async def create_item(query: Query):
    r = user_input(query.query)
    return {"text":r}

@app.post("/genratechattitle/") 
async def create_item(query: Query):
    r = get_conversational(query.query)
    return {"text":r}