import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)

BEST_MODEL = 'random_forest_best.pkl'

with open('transformer.obj', 'rb') as f:
    transform = pickle.load(f)

@app.route('/', methods=['POST','GET'])
def home():
    if request.method == 'POST':
        if 'csv' in request.files:
            csv_file = request.files['csv']
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=transform.all_cols)
            df = df.astype({'position_cat':int, 'page_views':int, 'fpl_value':float, 'fpl_points':int, 'region':float, 'new_foreign':int, 'age_cat':int, 'club_id':int, 'big_club':int, 'new_signing':int})
            name = str(request.form['name'])
            df.loc[0, 'position_cat'] = int(request.form['position'])
            df.loc[0, 'page_views'] = int(request.form['page_views'])
            df.loc[0, 'fpl_value'] = float(request.form['fpl_value'])
            df.loc[0, 'fpl_sel'] = str(request.form['fpl_sel']) + '%'
            df.loc[0, 'fpl_points'] = int(request.form['fpl_points'])
            df.loc[0, 'region'] = float(request.form['region'])
            df.loc[0, 'nationality'] = str(request.form['nation'])
            df.loc[0, 'club_id'] = int(request.form['club_id'])
            df.loc[0, 'new_foreign'] = int(request.form['new_foreign'])
            df.loc[0, 'age_cat'] = int(request.form['age_cat'])
            df.loc[0, 'big_club'] = int(request.form['big_club'])
            df.loc[0, 'new_signing'] = int(request.form['new_signing'])
        
        
        market_value = get_prediction(df)
        if df.shape[0] > 1:
            s = ''
            for i in range(df.shape[0]):
                s += f"Market Value for {df.loc[i, 'name']}: <b>{market_value[i]}</b><br>"
        else:
            s = f"Market Value for {name}: <b>{market_value}</b>"

        return s


    return render_template('index.html')


def get_prediction(df):
    X = transform.start(df)
    with open(BEST_MODEL, 'rb') as f:
        model = pickle.load(f)
    return np.squeeze(model.predict(X))


app.run()