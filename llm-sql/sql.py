from flask import Flask, request,  render_template
from transformers import AutoModelWithLMHead, AutoTokenizer

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-wikiSQL")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-wikiSQL")


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/translate', methods=['GET','POST'])
def translate():
    try:
        query = request.form.get('query')
        # Translate the query to SQL
        input_text = "translate English to SQL: %s " % query
        features = tokenizer([input_text], return_tensors='pt')
        output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'])
        sql_translation = tokenizer.decode(output[0])

        return render_template('index.html', sql_translation=sql_translation)

    except Exception as e:
        return render_template('index.html', sql_translation='Error: ' + str(e))

if __name__ == '__main__':
    app.run(debug=True)
