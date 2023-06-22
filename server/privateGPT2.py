from flask import Flask, jsonify, render_template, flash, redirect, url_for, Markup, request
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
import requests

app = Flask(__name__)
CORS(app)

load_dotenv()

embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_n_ctx = os.environ.get('MODEL_N_CTX')
llm = None


@app.route('/get_answer', methods=['POST'])
def get_answer():
    query = request.json
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            'You are helpful assistant, a large language model, answer as concisely as possible.'),
        HumanMessagePromptTemplate.from_template('{query}')
    ])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    if llm == None:
        return 'Model not downloaded', 400
    if query != None and query != '':
        answer = llm_chain.run({query: query})

        return jsonify(query=query, answer=answer)

    return 'Empty Query', 400


@app.route('/download_model', methods=['GET'])
def download_and_save():
    filename = request.args.get('model_name')  # Downloaded file name
    # Download url
    url = f'https://huggingface.co/{filename}/blob/main/pytorch_model-00002-of-00002.bin' if filename.find(
        '13b') > -1 else f'https://huggingface.co/{filename}/blob/main/pytorch_model-00003-of-00003.bin'
    models_folder = 'models'  # Specify the name of the folder inside the Flask app root

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    bytes_downloaded = 0
    file_path = f'{models_folder}/{filename}'
    if os.path.exists(file_path):
        return jsonify(response='Download completed')

    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=4096):
            file.write(chunk)
            bytes_downloaded += len(chunk)
            progress = round((bytes_downloaded / total_size) * 100, 2)
            print(f'Download Progress: {progress}%')
    global llm
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = LlamaCpp(
        model_path=file_path,
        n_gpu_layers=40,
        n_batch=512,
        callback_manager=callbacks,
        verbose=True,
    )
    return jsonify(response='Download completed')


if __name__ == '__main__':
    global model_name
    model_name = None
    app.run(host='0.0.0.0', debug=False)
