import os
from datasets import load_dataset, DatasetDict
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments
)

# ========== 路径配置 ==========
base_dir = "G:/PycharmProjects/NLP/BERT&GPT"
model_path = os.path.join(base_dir, "bert-base-chinese")
data_dir = os.path.join(base_dir, "chnsenticorp")

train_file = os.path.join(data_dir, "train/part.0")
dev_file = os.path.join(data_dir, "dev/part.0")
test_file = os.path.join(data_dir, "test/part.0")  # 可选

# ========== 加载数据 ==========
dataset = load_dataset("csv",
                       data_files={
                           "train": train_file,
                           "validation": dev_file,
                           "test": test_file if os.path.exists(test_file) else dev_file
                       },
                       delimiter="\t",
                       column_names=["text", "label"])

# ========== 加载本地模型和 tokenizer ==========
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# ========== Tokenize ==========
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_fn, batched=True)
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ========== 设置训练参数 ==========
training_args = TrainingArguments(
    output_dir=os.path.join(base_dir, "bert_output"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=os.path.join(base_dir, "logs"),
    logging_steps=50,
)

# ========== 初始化 Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# ========== 开始训练 ==========
trainer.train()

# ========== 保存模型 ==========
save_dir = os.path.join(base_dir, "bert_finetuned_chnsenticorp")
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"模型已保存至：{save_dir}")
