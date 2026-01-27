from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join(ds["text"])
open("data/wikitext2.txt","w",encoding="utf-8").write(text)
print("wrote", len(text), "chars")