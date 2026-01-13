# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:09:21 2023

@author: BANDASAB
"""
import os
from flask import Flask,request,jsonify
from werkzeug.utils import secure_filename
from Docu_quest import doc_splitter,store_embeddings,load_embeddings,Q_and_A,embeddings_
from summarize_book import summary_of_text
import openai
import shutil
FILE_DIR = "Files"
app = Flask(__name__)

###########################################################

def cleaning_task():
    shutil.rmtree(FILE_DIR)
    os.mkdir(FILE_DIR)

@app.route("/save_emb",methods = ["POST"])
def upload_files():
    cleaning_task()
    batch_id = request.form["batch_id"]
    files = request.files.getlist('file')
    for file in files:
        fn = secure_filename(file.filename)
        file.save(os.path.join(FILE_DIR,fn))
    store_embeddings(doc_splitter(FILE_DIR,chunk_size=2000), 
    embeddings_, 
    model_name='model_{}'.format(batch_id),)
    summary = summary_of_text(FILE_DIR)
    # print("SUMMARY",summary)
    return jsonify({"batch_id":batch_id,"summary":summary["summaries"],"rel_questions":summary["relevent_q"]})




##################################################### CHAT API ###############################################

@app.route("/chat",methods = ["POST"])
def load_emb_chat():
    req = request.get_json()
    question =  req["question"]
    batch_id = req["batch_id"]
    # file_list = request.form.getlist("file_names")
    file_names = req["file_names"]
    print("888888 filenames",file_names)
    file_names = [x.replace("   "," ") for x in file_names]
    # file_id_list = request.form.getlist("file_ids")
    file_id_list = req["file_ids"]
    dict_ = {"file_id_list":file_id_list,"file_name_list":file_names}
    # with open("user_data.json","r") as ud:
    #     user_data = json.load(ud)
    print("BATCH_ID",batch_id)
    model_name='model_{}'.format(batch_id)
    answer = Q_and_A(question,model_name)
    file_name_document = answer["source_documents"]
    # file_name_document = list(set(file_name_document))
    final_file_ids = []
    for i in file_name_document:
        # file_name_document = i.split("\\")[1]
        # i = i.replace("_"," ")
        print("file*****name******document",(i))
        file_id = file_id_list[file_names.index(i)]
        final_file_ids.append(file_id)
    # print("final_file_ids",final_file_ids)
    answer.update({"file_ids":final_file_ids})
    # final_file_ids = file_names.index(answer["source_documents"][0])
    return jsonify(answer)

##############################################################GET RELEVENT QUESTIONS ###############################################

@app.route("/get_relevent_questions",methods = ["POST"])
def get_relevent_questions():
    output = request.form["text_input"]
    relevent_q_prompt = "Give me the only three relevent questions that user might ask from the following text \n text:{}".format(output)
    # relevent_q_prompt = "Ask three quesions from the following text \n text:{}".format(output)
    response = openai.Completion.create(engine = "dev-taiaoai-td003",prompt = relevent_q_prompt,max_tokens=500)
    # print(output)
    # print(response)
    response = response["choices"][0]["text"]
    response = response.split("\n")[2:]
    return jsonify({"summary":output,"relevent_q":response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,port=8052)
    
    
    
