import tqdm.auto as tqdm
from transformers import AutoTokenizer
from data import load_plubmed
import pickle
import os

model_name = "microsoft/biogpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)


dataset = load_plubmed()
train_dataset = dataset["train"]


def format_dataset(examples):
    inputs = [f"Question: {q} " for q in examples["question"]]

    format_context = []
    for context in train_dataset["context"]:
        single_context = []
        for texts in context["contexts"]:
           single_context.append(texts)
        single_context = " ".join(single_context)
        format_context.append(single_context)

    outputs = [
        f"Context: {c} Answer: {a} Decision: {d}" for c,a,d in zip(format_context, examples["long_answer"], examples["final_decision"])
    ]
    texts = [inp+out for inp, out in zip(inputs, outputs)]

    return texts

formatted_dataset = format_dataset(train_dataset)

def tokenize_text(text_list, tokenizer):
    tokenized_texts = []
    for text in tqdm.tqdm(text_list):
        text = tokenizer.encode(text)
        tokenized_texts.append(text)
    return tokenized_texts

tokenized_texts = tokenize_text(text_list=formatted_dataset, tokenizer=tokenizer)

def save(data, dir="./data"):
    print(f"Saving data to {dir}")
    if not os.path.isdir(dir):
        os.makedirs(dir)

    data_path = f"{dir}/tokenized_text.pickle"

    if os.path.exists(data_path):
        print("Dataset exists, type R to replace or anything to skip")
        user_input = input("you:")
        if user_input == "R":
            with open(data_path, "wb") as f:
                pickle.dump(data, f)
            print("Data successfully replaced")
        else:
            print("Skipping...")

    else:
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
        print("Data saved successfully")

save(tokenized_texts)

