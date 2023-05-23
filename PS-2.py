#The objective of this repo is to fine tune a decoder model with the custom data set

"""Challenges - Understand the architcture of the backbone model
            Dataset preparation
            Performance comparison and metriccs"""


# In this project we are fintuning a small gpt2 model with finance article headlines 
#We have taken the data from https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?select=raw_partner_headlines.csv



"""Performance Metrics - 
1. BELU and ROGUE 
Belu is a precision focussed metric which calculates the n-grap overlap of the actual sentence and the generated sentence. On the other hand Rogue is Recall Based
2. Perplexity
Measure of the proabability of for a sentence produced by the model trained on a dataset. the lower the perplexity , the better is the model"""

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import math

dataset = load_dataset('csv', data_files= 'datasets/raw_partner_headlines.csv',split="train[:5000]")

dataset = dataset.train_test_split(test_size = 0.2)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

def preprocess_function(examples):
    return tokenizer([x for x in examples["headline"]])

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names)

block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

final_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

training_args = TrainingArguments(
    output_dir="P2.0-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    use_mps_device=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model('./P2.0-model')
tokenizer.save_pretrained("./P2.0-model/tokenizer/")

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")




