import pickle
from transformers import BioGptForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import os
import tqdm.auto as tqdm

model_name = "microsoft/biogpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BioGptForCausalLM.from_pretrained(model_name)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

def load(dir="./data/tokenized_text.pickle"):
    print(f"loading dataset from {dir}")
    with open(dir, "rb") as f:
        data = pickle.load(f)
    print("data loaded successfully")
    return data

loaded_data = load()

### Freeze all the layers and unfreeze the last 4 layers on the model
for param in model.parameters():
  param.requires_grad= False

# Unfreeze only the last 4 layers
for i in range(4):  # Adjust this number to unfreeze more/fewer layers
    for param in model.biogpt.layers[-(i+1)].parameters():
        param.requires_grad = True

test_frac=0.8
training_num = int(len(loaded_data)*test_frac)
training_data = loaded_data[:training_num]
testing_data = loaded_data[training_num:]

output_dir = "./model"
if os.path.exists(output_dir):
    pass
else:
    os.mkdir(output_dir)
logging_dir = "./logs"
if os.path.exists:
    pass
else:
    os.mkdir(logging_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10,
    eval_strategy="steps",
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    warmup_steps=5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=testing_data,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Define the output.pth path for saving the state dictionary
output_dir = "./model/biogpt_state_dict"

# Save the model's state_dict
torch.save(model.state_dict(), output_dir)

print(f"Model's state_dict saved to {output_dir}")
