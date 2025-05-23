import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载模型和分词器
model_path = "finetuned-bert-chinese-base"  # 模型路径，如果是微调后的模型，需要指定微调后的路径
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(
    model_path, 
    num_labels=2,  # 二分类问题：正面和负面
    output_attentions=False,
    output_hidden_states=False
)

# 设置为评估模式
model.eval()

# 待分类的句子
sentences = [
    "剧情拖沓冗长，中途几次差点睡着。",
    "食物份量十足，性价比超高，吃得很满足！"
]

# 对句子进行分词和编码
inputs = tokenizer(
    sentences, 
    padding=True, 
    truncation=True, 
    max_length=128, 
    return_tensors="pt"
)

# 模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs.logits
predictions = torch.argmax(logits, dim=1)

# 输出分类结果
label_map = {0: "负面", 1: "正面"}
for i, sentence in enumerate(sentences):
    print(f"句子: {sentence}")
    print(f"情感倾向: {label_map[predictions[i].item()]}\n")    