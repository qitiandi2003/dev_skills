from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "roberta-base-finetuned-jd-binary-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

texts = [
    "剧情拖沓冗长，中途几次差点睡着。",
    "食物份量十足，性价比超高，吃得很满足！"
]

for text in texts:
    result = classifier(text)
    print(f"【输入】：{text}")
    print(f"【分类】：{result[0]['label']}，置信度：{result[0]['score']:.4f}")
    print()
