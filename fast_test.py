from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline

model_dir = "./checkpoint-3682"
tokenizer = PegasusTokenizer.from_pretrained(model_dir)
model = PegasusForConditionalGeneration.from_pretrained(model_dir)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)
print(summarizer("A: Hi! B: Hello! A: Lunch today? B: Not Sure.", max_length=30, min_length=8, do_sample=False)[0]["summary_text"])
