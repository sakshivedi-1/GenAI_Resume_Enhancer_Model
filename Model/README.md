#  GenAI Resume Enhancer (Fine-tuned Flan-T5 Small)

This model fine-tunes `google/flan-t5-small` to enhance raw or basic resume lines into polished, recruiter-friendly statements — enabling GenAI-powered resume improvement tools and applications.

---

##  Model Objective

To generate professional, high-impact resume content from informal or shorthand lines using a generative transformer model.

> **Input:** Basic/informal resume phrase  
> **Output:** Enhanced, recruiter-ready line

---

## 📊 Dataset Used

- **Name:** [Synthetic Resume Dataset](https://huggingface.co/datasets/Sakshivedi/synthetic-resume-dataset)
- **Samples:** 50 handcrafted `input` → `output` pairs
- **Format:** JSONL with two keys per row: `input`, `output`

Example:
```json
{
  "input": "led a team of developers",
  "output": "Successfully led a team of software developers to deliver high-impact solutions on time."
}



##  How to use model

from transformers import pipeline

pipe = pipeline("text2text-generation", model="Sakshivedi/GenAI_Resume_Enhancer")

response = pipe("worked on data visualization tools")[0]['generated_text']
print(response)


Model Link: https://huggingface.co/Sakshivedi/GenAI_Resume_Enhancer
Dataset link: https://huggingface.co/datasets/Sakshivedi/synthetic-resume-dataset