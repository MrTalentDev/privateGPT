from flask import Flask, jsonify, render_template, flash, redirect, url_for, Markup, request
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
import os
import requests

app = Flask(__name__)
CORS(app)

load_dotenv()

llm = None


@app.route('/get_answer', methods=['POST'])
def get_answer():
    query = request.json
    if llm == None:
        return 'Model not downloaded', 400
    if query != None and query != '':
        template = """{question}"""

        prompt = PromptTemplate(
            template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        answer = llm_chain.run(query, callbacks=[
                               StreamingStdOutCallbackHandler()])

        return jsonify(query=query, answer=answer)

    return 'Empty Query', 400


@app.route('/download_model', methods=['GET'])
def download_and_save():
    full_name = request.args.get('model_name')  # Downloaded file name
    folder_name = full_name.split('/')[0]
    file_name = full_name.split('/')[1]
    # Download url
    url = f'https://huggingface.co/TheBloke/koala-13B-GGML/resolve/main/koala-13B.ggmlv3.q8_0.bin'
    print(url)
    models_folder = 'models'  # Specify the name of the folder inside the Flask app root

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    if not os.path.exists(f'{models_folder}/{folder_name}'):
        os.makedirs(f'{models_folder}/{folder_name}')
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    bytes_downloaded = 0
    file_path = f'{models_folder}/{folder_name}/{file_name}'
    if not os.path.exists(file_path):
        # return jsonify(response='Download completed')
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=4096):
                file.write(chunk)
                bytes_downloaded += len(chunk)
                progress = round((bytes_downloaded / total_size) * 100, 2)
                print(f'Download Progress: {progress}%')
    global llm
    callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
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
