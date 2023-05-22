#If you are given a set of instructions (domain knowledge, 
# up to 100 rules) and patient data, how do we create Q/A models 
# that answer questions based on the provided rules without any hardcoded Q/A or a pre-defined workflow.

#One of the Solution - In general any transformer based QA model requires atleast thousands of examples to perform well on any given dataset.But we have only 100 examples
                       # And the problem doesn't explicitly ask for any extractive or generative kind of solution, but with the recent advancements in generative solutions
                       # the ideal practice is use a decoder based model which can generate response given any question. Now the first thing that rings everybody mind 
                       # when dealing with less trained data is to go with few shots or zero shot model. Therfore the idea is to use a generative model which is trained on some medical data 
                       # which we further train it with patients data and then give some examples of questions and answers(rules) to generate more samples of questions and answers.


# Another Solution -  To create a chatbot like chatgpt, use reinforcement learning to finuetune the pretrained model on patients data


#Objective - This py file is the first version of data augmentation/text generation without further postprocessing of the output

import argparse
import logging

import numpy as np
import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename='PS1.log',
)
logger = logging.getLogger(__name__)

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_data(data_file_path,num_samples):
    with open(data_file_path) as rr:
        data = json.load(rr)
    return random.sample(data['questions'],num_samples)

def get_prompt(qas):
    description = "Each item in the following list contains a Question and the respective Answer."
    description = 'Generate more Questions like below.'
    prompt = (f"{description}\n"
            "Example : \n"
            f"Question: {qas[0]['question']}\n"
            f"Question:")
    
   

    """   prompt = (f"{description}\n"
            f"Question: {qas[0]['question']} (Answer: {qas[0]['answer']})\n"
            f"Question: {qas[1]['question']} (Answer: {qas[1]['answer']})\n"
            f"Question:")"""
    

    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="microsoft/biogpt",
        type=str,
    )
    parser.add_argument("--data_file_path", type=str, default="sample-medical-qa.json")
    parser.add_argument("--output_file_path", type=str, default="output-ps1.json")

    parser.add_argument("--num_input_samples", type=int, default=2)
    parser.add_argument("--total_input_samples", type=int, default=10)

    parser.add_argument("--num_output_samples", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args)

    data = load_data(args.data_file_path,num_samples = args.total_input_samples)
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path) 

    model_config = model.generation_config
    logging.info(str(model_config))

    prompt = get_prompt(random.sample(data,args.num_input_samples))
    logging.info(f' Prompt \n {prompt}')
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)


    logging.info(f"Settings ---- min_new_tokens= 50,max_length = 200")
    out = generator(prompt, num_return_sequences=5, do_sample=True,
                        min_new_tokens= 50,max_length = 200)

    """out = generator(prompt, num_return_sequences=3, do_sample=True,
                        temperature = 0.7,top_p=0.92,
                        min_length = 250)"""

    for o in out:
        o = o['generated_text'].replace(prompt,'')
        logging.info(o)


if __name__ == "__main__":
    main()