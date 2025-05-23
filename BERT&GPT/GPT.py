import torch
from transformers import GPT2LMHeadModel, BertTokenizer

# 模型和分词器本地路径
model_path = r"G:\PycharmProjects\NLP\BERT&GPT\gpt2-chinese-cluecorpussmall"

# 使用 BertTokenizer，因为这个中文 GPT2 模型用的是 vocab.txt 格式
tokenizer = BertTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# 要续写的输入文本
prompt = "假如我能隐身一天，我会"

# 将输入文本编码为模型输入
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 使用模型生成文本
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=45, #输出字数
        num_return_sequences=3,  # 输出三条续写
        temperature=0.8,  # 控制多样性
        top_k=50,
        top_p=0.95,
        do_sample=True,  # 启用随机采样
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

# 解码输出并打印
for i, sample_output in enumerate(output):
    text = tokenizer.decode(sample_output, skip_special_tokens=True)
    print(f"续写结果 {i+1}：\n{text}\n{'='*50}")
