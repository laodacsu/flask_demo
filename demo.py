#!/usr/bin/rnv python
# -*- coding:utf-8 -*-
# authoor:zjd time:2020/3/20

# main.py
from flask import Flask
from flask_restful import Resource,Api
# from disp_predict import *
from flask import request
import json
app = Flask(__name__)


@app.route('/')
def index():
    return 'hello world'

@app.route('/predict/poi/<string:id>/day/<int:predict_len>')
def get_poi_day_predict(id,predict_len=7):
    print(id)
    print('day')
    print(predict_len)
    return '1'


@app.route('/predict/poi/<string:id>/hour/<int:predict_len>')
def get_poi_hour_predict(id,predict_len=24):
    print(id)
    print('hour')
    print(predict_len)
    return '1'

@app.route('/parameter/poi/<string:id>')
def get_parameter_poi(id):
    return '1'

@app.route('/predict', methods = ['GET','POST'])
def predict():
    field = ['mac','poi','step','len']
    mac = request.values.get('mac')
    poi = request.values.get('poi')
    step = request.values.get('step')
    len = request.values.get('len')
    if len is None:
        print('null')
    print(mac)
    print(poi)
    print(step)
    print(len)
    return json.dumps(request.json)

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)