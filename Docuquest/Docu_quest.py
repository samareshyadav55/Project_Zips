# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:46:52 2023

@author: BANDASAB
"""
import os
import pickle
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain


####################### OPENAI KEYS ##############################
from dotenv import load_dotenv
load_dotenv()

openai.api_type = "azure"
openai.api_version = '2023-03-15-preview'
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

######################### EMBEDDINGS ###################################

def embeddings_():
    embeddings = OpenAIEmbeddings(engine = 'dev-embed-tai-aoai002', model="text-embedding-ada-002",chunk_size = 1)
    return embeddings

######################### CHUNCK SPLITTER #############################

def doc_splitter(file_path,chunk_size):
    
    loader = DirectoryLoader(f'{file_path}', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"],
                                                   chunk_size=chunk_size, 
                                                   chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

######################## STORE AND LOAD EMB Function ####################

def store_embeddings(docs, embeddings,model_name):
    
    vectorStore = FAISS.from_documents(docs, embeddings_())

    with open("Models/{}.pkl".format(model_name), "wb") as f:
        pickle.dump(vectorStore, f)
def load_embeddings(model_name):
    with open("Models/{}.pkl".format(model_name), "rb") as f:
        VectorStore = pickle.load(f)
    return VectorStore

########################## Q and A function ################################

def Q_and_A(query,model_name):
    db_openai_emb = load_embeddings(model_name)
    retriever = db_openai_emb.as_retriever(search_kwargs={"k": 2})
    #print("retriever**************",retriever)
    qa_chain_openai = RetrievalQA.from_chain_type(OpenAI(engine = "dev-taiaoai-td003"),
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True)
    #print("RETRIVAL OBJECT QA CHAIN CREATED")
    llm_response = qa_chain_openai(query)
    #print("LLM RESPONSE",llm_response)
    source_documents = []
    page_numbers = {}
    for source in llm_response["source_documents"]:
        doc_path = source.metadata["source"].split("\\")[1]
        source_documents.append(doc_path)
        # print('doc path',doc_path not in page_numbers.keys(),doc_path)
        if doc_path not in page_numbers.keys():
            page_numbers[doc_path] = [source.metadata["page"]]
        else:
            pagenum = source.metadata["page"]
            if pagenum not in page_numbers[doc_path]:
                page_numbers[doc_path].append(pagenum)
        # page_numbers.append(source.metadata["page"])
    print("Page_Numbers*****",page_numbers)
    final_result  = {"answer":llm_response["result"],"source_documents":list(page_numbers.keys()),"page_numbers_ref":page_numbers,"file_ids":[]}
    return final_result
  


