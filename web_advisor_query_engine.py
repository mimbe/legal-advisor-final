from flask import Flask, request, render_template_string
import faiss
import numpy as np
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index and chunk metadata
faiss_index = faiss.read_index("data/faiss_index.index")
with open("data/chunk_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

HTML_TEMPLATE = """
<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>مشاور حقوقی قرارداد پیمانکاری</title>
  <style>
    body {
      font-family: Vazirmatn, Tahoma, sans-serif;
      background-color: #f4f4f4;
      padding: 40px;
      color: #333;
      max-width: 800px;
      margin: auto;
    }
    h2 {
      color: #005a8d;
    }
    textarea {
      width: 100%;
      padding: 10px;
      font-size: 16px;
      border: 1px solid #ccc;
      border-radius: 5px;
    }
    input[type="submit"] {
      background-color: #007bff;
      color: white;
      padding: 10px 20px;
      border: none;
      border-radius: 5px;
      font-size: 16px;
      cursor: pointer;
    }
    input[type="submit"]:hover {
      background-color: #0056b3;
    }
    .answer-box {
      background: white;
      border: 1px solid #ddd;
      padding: 20px;
      border-radius: 5px;
      margin-top: 30px;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <h2>🧾 مشاور حقوقی قرارداد پیمانکاری</h2>
  <form method=post>
    <label for="question">پرسش خود را وارد کنید:</label><br><br>
    <textarea name=question rows=5>{{ question }}</textarea><br><br>
    <input type=submit value="پرسیدن">
  </form>

  {% if answer %}
    <div class="answer-box">
      <h3>✅ پاسخ مشاور حقوقی:</h3>
      <div>{{ answer }}</div>
    </div>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def ask():
    question = ""
    answer = None

    if request.method == "POST":
        question = request.form["question"]
        if question.strip():
            question_embedding = client.embeddings.create(
                input=question,
                model="text-embedding-ada-002"
            ).data[0].embedding

            query_vector = np.array([question_embedding]).astype("float32")
            distances, indices = faiss_index.search(query_vector, 5)
            matched_chunks = [chunks[i] for i in indices[0]]

            context_text = "\n\n".join([f"{c['title']}:\n{c['content']}" for c in matched_chunks])

            prompt = f"""
شما یک مشاور حقوقی تخصصی در قراردادهای پیمانکاری ایران هستید.

با استناد به مواد زیر از شرایط عمومی پیمان پاسخ دهید
توجه: شماره موادی که به آن استناد می کنید را حتما ذکر کنید


📘 متن مواد قانونی از شرایط عمومی پیمان:
{context_text}

❓ سؤال:
{question}
❗ پاسخ را با بله یا خیر شروع نکن. به متن مواد قانونی مراجعه کن و در اگر توانستید در جملات پایانی پاسخی که ارائه میکنید از بله یا خیر استفاده کن 
📌 اگر موضوع خارج از حوزه شرایط عمومی پیمان بود، پاسخ را فقط با این جمله شروع کنید:
«این موضوع خارج از حوزه تخصص من به عنوان یک مشاور حقوقی پیمان است.»  
و **به هیچ عنوان با واژه‌هایی مانند «بله»، «خیر»، «درست است» یا «می‌توان گفت» آغاز نکنید.**
پاسخ را شفاف، حقوقی، با جمله‌بندی دقیق، رسمی و ساده و مستند به مواد قانونی موجود در متن مواد قانونی شرایط عمومی پیمان ارائه بده. از تکرار بی‌مورد و ترجمه ماشینی پرهیز کن.
"""

            response = client.chat.completions.create(
               model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()

    return render_template_string(HTML_TEMPLATE, question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True, port=5000)