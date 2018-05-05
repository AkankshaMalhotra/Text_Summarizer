from flask import Flask, request, render_template

from load_model import load
from text_summarizer import TextSummarizer

term_document_matrix_inv, index_tdm = load()
app = Flask(__name__)


@app.route('/success', methods=['POST', 'GET'])
def success():
    if request.method == 'POST':
        result = request.form
        text = result["text"]
        flag = result["flag"]
    else:
        text = request.args.get('text')
        flag = request.args.get('flag')
    summarize = TextSummarizer(text, flag, term_document_matrix_inv, index_tdm)
    summary = summarize.get_summary
    text = text.replace("\\n", "\n")
    result = {"text": text, "summary": summary}
    return render_template("success.html", success=result)


@app.route('/summarizer')
def summarizer():
    return render_template('summarizer.html')


if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
