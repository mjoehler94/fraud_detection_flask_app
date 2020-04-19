from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import os
import pickle
import modelapi

# global values that should only be loaded once --------
with open("data/good_example.json", 'r') as f:
    good_json = f.read()

with open("data/fraud_example.json", 'r') as f:
    fraud_json = f.read()

model = modelapi.load_model()
# end of global values ---------------------------------

app = Flask(__name__)


class JsonForm(Form):
    # data_string = TextAreaField('', [validators.DataRequired()])
    json_string = TextAreaField('', [validators.DataRequired()])

@app.route('/')
def index():
    form = JsonForm(request.form)
    return render_template('jsonform.html',
                           form=form,
                           good_json=good_json,
                           fraud_json=fraud_json)


@app.route('/results', methods=["POST"])
def results():
    form = JsonForm(request.form)
    if request.method == 'POST' and form.validate():
        json_data = request.form['json_string']
        result, proba = modelapi.make_prediction(model, json_data)

        return render_template('results.html',
                               json_string=json_data,
                               result=result,
                               proba=proba)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
