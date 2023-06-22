from flask import Flask, jsonify, render_template, flash, redirect, url_for, Markup, request
from flask_cors import CORS
from dotenv import load_dotenv
import os

from modal import Image, Stub, gpu, method
from transformers import LlamaForCausalLM, LlamaTokenizer


app = Flask(__name__)
CORS(app)

load_dotenv()

embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
persist_directory = os.environ.get('PERSIST_DIRECTORY')
n_gpu_layers = 40
n_batch = 512

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
llm = None

GPT4Model = 'GPT4All'
Falcon7BModel = 'Falcon-7B'
Koala13BModel = 'Koala-13b'
Vicuna13BModel = 'Vicuna-13b'


def download_model():
    LlamaForCausalLM.from_pretrained(model_name)
    LlamaTokenizer.from_pretrained(model_name)


image = (
    # Python 3.11+ not yet supported for torch.compile
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate~=0.18.0",
        "transformers~=4.28.1",
        "torch~=2.0.0",
        "sentencepiece~=0.1.97",
    )
    .run_function(download_model)
)
stub = Stub(name="example-open-llama", image=image)


@app.route('/download_model', methods=['GET'])
def download_and_save():
    args = request.args
    global model_name
    model_name = args.get('model_name')
    print('Downloading ' + args.get('model_name'))
    global stub
    stub = Stub(name="example-open-llama", image=image)

    return jsonify(response='Download completed')


@app.route('/get_answer', methods=['POST'])
def get_answer():
    query = request.json

    if model_name == None or model_name == '':
        return 'Model is not downloaded', 400
    if query != None and query != '':
        model = OpenLlamaModel()
        with stub.run():
           answer = model.generate.call(
               query,
               top_p=0.75,
               top_k=40,
               num_beams=1,
               temperature=0.1,
               do_sample=True,
           )

        return jsonify(query=query, answer=answer)

    return 'Empty Query', 400


@stub.cls(gpu=gpu.A100(memory=20))
class OpenLlamaModel:
    def __enter__(self):
        import torch
        from transformers import LlamaForCausalLM, LlamaTokenizer

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer.bos_token_id = 1

        model.eval()
        self.model = torch.compile(model)
        self.device = "cuda"

    @method()
    def generate(
        self,
        input,
        max_new_tokens=128,
        **kwargs
    ):
        import torch
        from transformers import GenerationConfig

        inputs = self.tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        generation_config = GenerationConfig(**kwargs)
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        # print(f"\033[96m{input}\033[0m")
        print(output.split(input)[1].strip())
        return output.split(input)[1].strip()


if __name__ == '__main__':
    global model_name
    model_name = None
    app.run(host='0.0.0.0', debug=False)
