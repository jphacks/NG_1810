# -*- coding: utf-8 -*-

from bottle import route, get, post, run, HTTPResponse, request, static_file
from bottle import template, redirect
import json
from nn_model import predict_onm
from datetime import datetime
import csv
import random

@route('/', method='GET')
def index():
    return template('load')


def get_save_path():
    path_dir = "./static/img/"
    return path_dir


@route('/api/upload-image', method='POST')
def upload_image():
    nowtime = datetime.now().strftime("%Y%m%d%H%M%S")
    upload = request.files.get('upload')

    if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return HTTPResponse('File extension not allowed!')

    save_path = get_save_path()
    upload.filename = nowtime + '_' + upload.filename
    upload.save(save_path)

    body = json.dumps({
        'message': 'OK',
        'data': upload.filename
    })

    redirect("/onoma/" + upload.filename)


@get('/onoma/:name')
def get_image(name):
    onm = predict_onm(name)
    dic_type = {"drop": "0", "flow": "1", "rain": "2", "soil": "3", "wave": "4", "wind": "5"}
    word_list = []
    onm_type = dic_type[onm]
    with open('word.csv', 'r', encoding="utf8") as f:
        reader = csv.reader(f)
        for line in reader:
            if line[0] == onm_type:
                word_list.append(line[1])
    onm = random.choice(word_list)
    return template('draw', name=name, onm=onm)


@route('/static/<file_path:path>')
def static(file_path):
    return static_file(file_path, root='./static')


if __name__ == '__main__':
    run(host='localhost', port=8080, debug=True)
