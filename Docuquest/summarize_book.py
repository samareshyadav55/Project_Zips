# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:03:41 2023

@author: BANDASAB
"""


from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model
from langchain.chat_models import ChatOpenAI

# Embedding Support
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain

# Data Science
import numpy as np
from sklearn.cluster import KMeans

from langchain.document_loaders import DirectoryLoader,PyPDFLoader
# from langchain.document_loaders import 
# Load the book
from langchain import OpenAI
import openai,os

############################## KEYS ##############################
from dotenv import load_dotenv
load_dotenv()

openai.api_type = "azure"
openai.api_version = '2023-03-15-preview'
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(engine = "dev-chat-tai-aoai",temperature=0)

#################################SUMMARY FUNCTION ##########################

def summary_of_text(file_path):

    loader = DirectoryLoader(f'{file_path}', glob="./*.pdf", loader_cls=PyPDFLoader)
    pages = loader.load()
    
    print("LOADING TEXT")
    text = ""
    
    for page in pages:
        text += page.page_content
    
    text = text.replace('\t', ' ')
    print("SPLITTING TEXT INTO CHUNCKS")
    num_tokens = llm.get_num_tokens(text)
    
    print (f"This book has {num_tokens} tokens in it")
    if num_tokens>=2000 and num_tokens<10000:
        chunk_size = 700
        chunk_overlap = 100
    elif num_tokens>=10000 and num_tokens<30000:
        chunk_size = 1100
        chunk_overlap =100
    elif num_tokens>=30000 and num_tokens<50000:
        chunk_size = 2000
        chunk_overlap = 200
    elif num_tokens>=50000 and num_tokens<100000:
        chunk_size = 7000
        chunk_overlap = 1000
    elif num_tokens>=100000:
        chunk_size = 10000
        chunk_overlap=3000
    else:
        chunk_size = 500
        chunk_overlap = 50
    
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    from sklearn.metrics import silhouette_score
    docs = text_splitter.create_documents([text])
    print("LEngth of docs",len(docs))
    embeddings = OpenAIEmbeddings(engine = 'dev-embed-tai-aoai002', model="text-embedding-ada-002",chunk_size = 1)
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    silhouette_scores = []
    best_score = -1
    best_k = 2
    for k in range(2, len(docs)-1):
        kmeans = KMeans(n_clusters=k, max_iter=300,  random_state=0)
        cluster_labels = kmeans.fit_predict(vectors)
        print("CLUSTER LABELS",cluster_labels)
        silhouette_avg = silhouette_score(vectors, cluster_labels)
        silhouette_scores.append(silhouette_avg) 
        print("BEST_CLUSTER Value before IF",best_k)
        if silhouette_avg > best_score:
          best_score = silhouette_avg
          best_k = k
    print("BEST CLUSTER VALUE",best_k)
    kmeans = KMeans(n_clusters=best_k, random_state=42).fit(vectors)
    closest_indices = []
    
    # Loop through the number of clusters you have
    for i in range(best_k):
    
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        print(distances)
    
        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)
        print(closest_index)
    
        # Append that position to your closest indices list
        closest_indices.append(closest_index)
    selected_indices = sorted(closest_indices)
    selected_indices
    from langchain import PromptTemplate
    map_prompt = """
    You will be given a single batch of summary. This section will be enclosed in triple backticks (```)
    Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
    Your response should be at least three paragraphs and fully encompass what was said in the passage.
    
    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(llm=llm,
                                 chain_type="stuff",
                                 prompt=map_prompt_template)
    selected_docs = [docs[doc] for doc in selected_indices]
    # Make an empty list to hold your summaries
    summary_list = []
    
    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
    
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])
    
        # Append that summary to your list
        summary_list.append(chunk_summary)
    
        print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")
    summaries = "\n".join(summary_list)
    
    # Convert it back to a document
    summaries = Document(page_content=summaries)
    
    print (f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")
    combine_prompt = """
    You will be given a batch of summaries. Your goal is to give a overall summary from batch of summaires. what happened in the story.
    The reader should be able to grasp what happened in the book.
    
    ```{text}```
    VERBOSE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    reduce_chain = load_summarize_chain(llm=llm,
                                 chain_type="stuff",
                                 prompt=combine_prompt_template,
    #                              verbose=True # Set this to true if you want to see the inner workings
                                       )
    output = reduce_chain.run([summaries])
    relevent_q_prompt = "Give me the only three relevent questions that user might ask from the following text \n text:{}".format(output)
    # relevent_q_prompt = "Ask three quesions from the following text \n text:{}".format(output)
    response = openai.Completion.create(engine = "dev-taiaoai-td003",prompt = relevent_q_prompt,max_tokens=500)
    print(output)
    print(response)
    response = response["choices"][0]["text"]
    response = response.split("\n")[2:]
    return {"summaries":output,"relevent_q":response}
# summary_of_text("Files")
