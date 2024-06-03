# -*- coding: utf-8 -*-
from flask import Flask, render_template
from flask_wtf import FlaskForm, CSRFProtect
from wtforms.validators import DataRequired, Length
from wtforms.fields import *
from flask_bootstrap import Bootstrap5, SwitchField
from input_processor import processInput, computeDistances
import csv

app = Flask(__name__)
app.secret_key = 'dev'

# set default button sytle and size, will be overwritten by macro parameters
app.config['BOOTSTRAP_BTN_STYLE'] = 'primary'
app.config['BOOTSTRAP_BTN_SIZE'] = 'sm'

# set default icon title of table actions
app.config['BOOTSTRAP_TABLE_VIEW_TITLE'] = 'Read'
app.config['BOOTSTRAP_TABLE_EDIT_TITLE'] = 'Update'
app.config['BOOTSTRAP_TABLE_DELETE_TITLE'] = 'Remove'
app.config['BOOTSTRAP_TABLE_NEW_TITLE'] = 'Create'

bootstrap = Bootstrap5(app)
csrf = CSRFProtect(app)

class ButtonForm(FlaskForm):
    confirm = SwitchField('Confirmation')
    submit = SubmitField()
    delete = SubmitField()
    cancel = SubmitField()

class SynopsisForm(FlaskForm):
    synopsis = TextAreaField('', validators=[DataRequired(), Length(1, 2000)])
    submit = SubmitField()

def csv_to_dict_list(file_path):
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        dict_list = [dict(row) for row in reader]
    return dict_list

@app.route('/', methods=['GET', 'POST'])
def index():
    form = SynopsisForm()
    if form.validate_on_submit():
        synopsis_data = form.synopsis.data

        # Add call to model here
        X_lsa, cluster = processInput(synopsis_data)
        # clustered_movies = pd.read_csv('clustered_movies.csv')
        recommendations = computeDistances(X_lsa, cluster) # clustered_movies.loc[clustered_movies['cluster'] == cluster].sort_values('rating', ascending=False).head(10)
        data = recommendations.to_dict('records') # csv_to_dict_list('clustered_movies.csv')

        return render_template(
            'index.html',
            form=form,
            button_form=ButtonForm(),
            rows=data,
        )
    return render_template(
        'index.html',
        form=form,
        button_form=ButtonForm(),
    )

if __name__ == '__main__':
    app.run(host="localhost", port=3000, debug=True)