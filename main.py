# -*- coding: utf-8 -*-

from bottle import route, get, post, run, HTTPResponse, request, static_file
from bottle import template, redirect
import json
from datetime import datetime


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
    return template('draw', name=name)


@route('/static/<file_path:path>')
def static(file_path):
    return static_file(file_path, root='./static')


if __name__ == '__main__':
    run(host='localhost', port=8080, debug=True)
