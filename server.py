from flask import Flask, render_template, request
from model.punctuate import punctuate
import torch

app = Flask(__name__)


def load_model():
    model = torch.load("model/punctuation_model.pth", map_location=torch.device('cpu'))
    model.eval()
    tokenizer = torch.load("model/tokenizer.pth")
    tag_values = torch.load("model/tag_values.pth")
    return model, tokenizer, tag_values


model, tokenizer, tag_values = load_model()


@app.route('/')
def upload():
    return render_template("index.html")


@app.route('/index', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        return render_template("index.html", punc=punctuate(text, model, tokenizer, tag_values))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
