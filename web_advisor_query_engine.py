from flask import Flask, request, render_template_string
import faiss
import numpy as np
import json
import os
import requests
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# مسیرها در سطح فایل
FAISS_PATH = "data/faiss_index.index"
FAISS_URL = "https://github.com/mimbe/legal-advisor-final/releases/download/v1/faiss_index.index"

# بررسی و دانلود فایل FAISS در صورت نیاز و بارگذاری آن
def load_faiss_index():
    if not os.path.exists(FAISS_PATH):
        print("📥 Downloading FAISS index from GitHub Releases...")
        os.makedirs("data", exist_ok=True)
        response = requests.get(FAISS_URL, timeout=10)
        if response.status_code == 200:
            with open(FAISS_PATH, "wb") as f:
                f.write(response.content)
        else:
            raise RuntimeError(f"❌ Failed to download FAISS index. Status code: {response.status_code}")

    return faiss.read_index(FAISS_PATH)

# متغیرها برای FAISS index و chunks
faiss_index = None
chunks = None

def init_faiss():
    global faiss_index, chunks
    faiss_index = load_faiss_index()
    with open("data/chunk_metadata.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

init_faiss()

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

با استناد به مواد زیر از شرایط عمومی پیمان پاسخ دهید.
توجه: شماره موادی که به آن استناد می‌کنید را حتماً ذکر کنید.

📘 متن مواد قانونی از شرایط عمومی پیمان:
{context_text}

❓ سؤال:
{question}

📌 فقط اگر اطلاعات دقیق و مرتبط در شرایط عمومی پیمان وجود دارد، پاسخ دهید. در غیر این صورت، فقط بنویسید: «این موضوع خارج از حوزه تخصص من است.»

پاسخ را شفاف، حقوقی، دقیق و مستند به مواد قانونی ارائه دهید. از جمله‌های دوپهلو و ترجمه ماشینی پرهیز کنید.
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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
