import fitz
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text.strip().replace("\n", " ")

def chunk_text(text, chunk_size=450):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def summarize_text(text):
    text = text.replace('\n',' ').strip()
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
      input_text = "summarize: " + chunk
      input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

      summary_ids = model.generate(
          input_ids,
          max_length=500,
          min_length=100,
          length_penalty=2.0,
          num_beams=4,
          early_stopping=True,
          do_sample = True
      )
      summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
      summaries.append(summary)

    # output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # return output
    return ' '.join(summaries)
