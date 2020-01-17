import os

from flask import Flask, render_template, flash, redirect, url_for, request, send_from_directory, session
import requests
from config import *
import random
from preprocess import couplet_work

app = Flask(__name__, static_folder='static')
app.secret_key = os.getenv('SECRET_KEY', 'secret string')
app.config['UPLOAD_FOLDER'] = "./static/uploaded_files/"

font_dict = {}

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/', methods=['GET', 'POST'])
def welcome():
    return render_template('welcome.html')

@app.route('/show_couplet', methods=['GET', 'POST'])
def show_couplet(**kwargs):
    if request.method == 'POST':
        if 'myfont' not in session:
            session['myfont'] = {}
        font_id = request.form.get('font')
        first_line = request.form.get('first_line')
        second_line = request.form.get('second_line')
        horizontal_scroll = request.form.get('horizontal_scroll')
        if len(first_line) != up_num or len(second_line) != down_num or len(horizontal_scroll) != row_num:
            # print("length of couplet not correct")
            return redirect(url_for('.main_page_couplet'))
        target_text = first_line + second_line + horizontal_scroll + "福"
        # print(font_id)
        # print(len(target_text), target_text)

        req = requests.post("http://"+ server_address +"/get_char_img_list", 
                            data={"token":font_id, "string":target_text})
        # print("receive response__")
        # print(req.text)
        if req.status_code != 200:
            return redirect(url_for('.main_page_couplet'))

        img_url_list = req.json()['img_url_list']
        while(len(img_url_list)<text_total_len):
            img_url_list.append(str(img_url_list[-1]))

        calli_work = couplet_work(img_url_list)
        if 'myid' not in session:
            session['myid'] = str(random.randint(0, 1000000000))
            font_dict[session['myid']] = {}
        calli_work_path = work_url + str(session['myid']) + "calli" + str(random.randint(0, 1000000000)) + ".png"
        session["work_path"] = calli_work_path
        calli_work.save(calli_work_path)
        return render_template('show.html', work_image_url=calli_work_path)
    
    if "work_path" not in session:
        render_template('show.html')
    return render_template('show.html', work_image_url=session["work_path"])
    # return render_template('show.html', work_image_url="static/images/calli_work.png")

@app.route('/show_template', methods=['GET', 'POST'])
def show_template(**kwargs):
    if request.method == 'POST':
        if 'myfont' not in session:
            session['myfont'] = {}
        font_id = request.form.get('font')
        template_id = request.form.get('couplet_group')
        target_text = template_list[template_id] + "福"
        # print(font_id)
        # print(len(target_text), target_text)

        req = requests.post("http://"+ server_address +"/get_char_img_list", 
                            data={"token":font_id, "string":target_text})
        # print("receive response__")
        # print(req.text)
        if req.status_code != 200:
            return redirect(url_for('.main_page_template'))

        img_url_list = req.json()['img_url_list']
        calli_work = couplet_work(img_url_list)
        if 'myid' not in session:
            session['myid'] = str(random.randint(0, 1000000000))
            font_dict[session['myid']] = {}
        calli_work_path = work_url + str(session['myid']) + "calli" + str(random.randint(0, 1000000000)) + ".png"
        session["work_path"] = calli_work_path
        calli_work.save(calli_work_path)
        return render_template('show.html', work_image_url=calli_work_path)
    
    if "work_path" not in session:
        render_template('show.html')
    return render_template('show.html', work_image_url=session["work_path"])
    # return render_template('show.html', work_image_url="static/images/calli_work.png")

@app.route('/show_demo', methods=['GET', 'POST'])
def show_demo(**kwargs):
    return render_template('show.html', work_image_url="static/images/calli_work.png")

@app.route('/operation/couplet', methods=['GET', 'POST'])
def main_page_couplet(**kwargs):
    if 'myid' not in session:
        session['myid'] = str(random.randint(0, 1000000000))
        font_dict[session['myid']] = {}
    elif session['myid'] not in font_dict:
        font_dict[session['myid']] = {}
    
    font_list = [(token, name) for token, name in font_dict[session['myid']].items()]
    for id, name in predefined_font_dict.items():
        font_list.append((id, name))

    return render_template('main_couplet.html', font_list=font_list)

@app.route('/operation/create', methods=['GET', 'POST'])
def main_page_create(**kwargs):
    if 'myid' not in session:
        session['myid'] = str(random.randint(0, 1000000000))
        font_dict[session['myid']] = {}
    elif session['myid'] not in font_dict:
        font_dict[session['myid']] = {}
    
    font_list = [(token, name) for token, name in font_dict[session['myid']].items()]

    return render_template('main_create.html', font_list=font_list)

@app.route('/operation/template', methods=['GET', 'POST'])
def main_page_template(**kwargs):
    if 'myid' not in session:
        session['myid'] = str(random.randint(0, 1000000000))
        font_dict[session['myid']] = {}
    elif session['myid'] not in font_dict:
        font_dict[session['myid']] = {}
    
    font_list = []
    for id, name in predefined_font_dict.items():
        font_list.append((id, name))
    font_list += [(token, name) for token, name in font_dict[session['myid']].items()]

    return render_template('main_template.html', font_list=font_list, couplet_list=couplet_list)


@app.route('/handwritingsubmit', methods=['POST'])
def handwriting_submit():
    if request.method == 'POST':
        if 'handw_image' not in request.files:
            # print("handw_image not in request.files")
            return redirect(url_for(".main_page_create"))
        handw_image = request.files.get('handw_image')
        font_name = request.form.get('font_name', "")
        if font_name == "" or not handw_image:
            return redirect(url_for(".main_page_create"))

        filename = handw_image.filename
        # print(font_name, filename)
        handw_image.save(os.path.join('./static/uploaded_files/', filename))

        # file_url = url_for('uploaded_file', filename=filename)
        image_file = {'img': open('./static/uploaded_files/'+filename, 'rb')}
        req = requests.post("http://"+ server_address +"/new_font", 
                                files=image_file)
        # print("receive response__")
        # print(req.text)
        if req.status_code != 200:
            # print("bad response")
            return redirect(url_for('.main_page_create'))
        font_token = req.json()['token']
        # print(font_token)

        if 'myid' not in session:
            session['myid'] = str(random.randint(0, 1000000000))
            font_dict[session['myid']] = {}
        elif session['myid'] not in font_dict:
            font_dict[session['myid']] = {}
        
        font_dict[session['myid']][font_token] = font_name
        ocr_image_url = "http://"+ server_address + "/output/" + font_token + "/rendered.jpg"
        return render_template('ocr_result.html', ocr_image_url=ocr_image_url)

if __name__ == '__main__':
    if IF_TEST:
        app.run(debug=True)
    else:
        app.run(debug=False, host=server_ip, port=server_port)
    