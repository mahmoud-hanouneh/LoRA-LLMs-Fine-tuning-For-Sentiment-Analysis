from datasets import load_dataset
from transformers import AutoTokenizer

from transformers import LlamaForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType


dataset = load_dataset("MohammadOthman/mo-customer-support-tweets-945k")

# Load the tokenizer  for the LLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Add a special padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 


def tokenize_function(examples):
    # Tokenize inputs (customer inquiries)
    inputs = tokenizer(
        examples['input'], padding="max_length", truncation=True, max_length=512
    )

    # Tokenize outputs (customer responses) to use as labels
    outputs = tokenizer(
        examples['output'], padding="max_length", truncation=True, max_length=512
    )

    # Ensure that labels are the tokenized responses
    inputs['labels'] = outputs['input_ids']

    return inputs


# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", 
    load_in_8bit=True, 

    device_map="auto"
)


# LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  
    r=16,                          
    lora_alpha=32,                
    lora_dropout=0.1,            
    target_modules=["q_proj", "v_proj"]  
)

# Apply Config LoRA to the model
model = get_peft_model(model, peft_config)


# Add token to the model 
model.resize_token_embeddings(len(tokenizer))


# Training arguments
training_args = TrainingArguments(
    output_dir="lora-llama2-customer-support",  
    # per_device_train_batch_size=2,
    auto_find_batch_size=True,              
    gradient_accumulation_steps=16,            
    num_train_epochs=2,                         
    learning_rate=2e-4,                         
    fp16=True,                                  
    logging_steps=10,                           
    save_steps=1000,                            
    save_total_limit=2,                         
    optim="adamw_torch"                         
)



data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"], 
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("lora-llama2-customer-support")
tokenizer.save_pretrained("lora-llama2-customer-support")