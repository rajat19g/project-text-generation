## Inferencing ...
# !pip install torch torchvision torchaudio transformers fastapi uvicorn

import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
#import nest_asyncio

app = FastAPI()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

class Input_Text(BaseModel):
    _id: str
    text: str

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("./P2.0-model/tokenizer")
model = AutoModelForCausalLM.from_pretrained("./P2.0-model/")
tokenizer.pad_token = tokenizer.eos_token

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/gen_text/")
async def create_item(text_input:Input_Text):

    prompt = text_input.text
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=10, do_sample=True, top_k=50, top_p=0.95)

    text_input.text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return text_input
    

# Run this file on command prompt with "uvicorn p2_infer:app --reload" command. Here p2_infer is the file name and app is the object we created in this file

# Go to http://127.0.0.1:8000/docs to access a swagger like UI to access the endpoimts
# docker build --platform=linux/amd64 -t gcr.io/terraform-gke-openvino/text-gen-amd:base . 
#docker build -t gcr.io/terraform-gke-openvino/text-gen:base .     --> gcr.io/<project ID>/<name> Specific to how GKE wants the image name 
#docker run -it -p 80:80 gcr.io/terraform-gke-openvino/text-gen-amd:base
#docker push gcr.io/terraform-gke-openvino/text-gen-amd:base

#gcloud container clusters get-credentials text-gen-cluster \
 #   --region us-central1

"""
kubectl apply -f build.yaml
 kubectl logs deployment/<name-of-deployment> # logs of deployment
kubectl logs -f deployment/yolo
kubectl logs -f deployment/<name-of-deployment> # follow logs
kubectl expose deployment yolo --type LoadBalancer --port 80 --target-port 80
 """

#curl -v http://localhost:50/items/1