## Inferencing ...

import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

class Input_Text(BaseModel):
    _id: str
    text: str

device = torch.device("mps")

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
