# -*- coding: utf-8 -*-
from enum import Enum
from flask import Flask, render_template, request, flash, redirect, url_for
from markupsafe import Markup
from flask_wtf import FlaskForm, CSRFProtect
from wtforms.validators import DataRequired, Length, Regexp
from wtforms.fields import *
from flask_bootstrap import Bootstrap5, SwitchField
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'dev'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

# set default button sytle and size, will be overwritten by macro parameters
app.config['BOOTSTRAP_BTN_STYLE'] = 'primary'
app.config['BOOTSTRAP_BTN_SIZE'] = 'sm'

# set default icon title of table actions
app.config['BOOTSTRAP_TABLE_VIEW_TITLE'] = 'Read'
app.config['BOOTSTRAP_TABLE_EDIT_TITLE'] = 'Update'
app.config['BOOTSTRAP_TABLE_DELETE_TITLE'] = 'Remove'
app.config['BOOTSTRAP_TABLE_NEW_TITLE'] = 'Create'

bootstrap = Bootstrap5(app)
db = SQLAlchemy(app)
csrf = CSRFProtect(app)

class ButtonForm(FlaskForm):
    confirm = SwitchField('Confirmation')
    submit = SubmitField()
    delete = SubmitField()
    cancel = SubmitField()

class SynopsisForm(FlaskForm):
    synopsis = TextAreaField('Synopsis', validators=[DataRequired(), Length(1, 500)])
    submit = SubmitField()

@app.route('/', methods=['GET', 'POST'])
def index():
    form = SynopsisForm()
    if form.validate_on_submit():
        synopsis_data = form.synopsis.data

        # Add call to model here

        return render_template(
            'index.html',
            form=form,
            button_form=ButtonForm(), result=synopsis_data
        )
    return render_template(
        'index.html',
        form=form,
        button_form=ButtonForm(),
    )

if __name__ == '__main__':
    app.run(debug=True)