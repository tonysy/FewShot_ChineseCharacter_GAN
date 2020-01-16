import os
import imghdr
import secrets
from flask import Flask, request, abort, send_from_directory, url_for
from werkzeug.utils import secure_filename

from pre_process import is_kanji
from doctext import render_doc_text

app = Flask(__name__)

app.config.update(
    dict(
        INPUT_PATH=os.path.join(app.root_path, "input"),
        OUTPUT_PATH=os.path.join(app.root_path, "output"),
        ALLOWED_IMG_FORMATS={"jpeg", "png"},
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        TOKEN_LENGTH=16,
        MAX_QUERY_LENGTH=32,
    )
)


@app.route("/new_font", methods=["GET", "POST"])
def new_font():
    if request.method == "POST":
        img = request.files.get("img")
        if img is None:
            abort(400)
        img_format = imghdr.what(img)
        if img_format not in app.config["ALLOWED_IMG_FORMATS"]:
            abort(415)

        token = secrets.token_hex(app.config["TOKEN_LENGTH"])
        filename = "{}.{}".format(token, img_format)
        file_path = os.path.join(app.config["INPUT_PATH"], filename)
        img.save(file_path)

        out_path = os.path.join(app.config["OUTPUT_PATH"], token)
        os.mkdir(out_path)
        render_doc_text(file_path, out_path)

        return {"token": token}

    return """
    <!doctype html>
    <title>Upload new Image</title>
    <h1>Upload new Image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=img>
      <input type=submit value=Upload>
    </form>
    """


@app.route("/get_char_img_list", methods=["GET", "POST"])
def get_char_img_list():
    if request.method == "POST":
        token = request.form.get("token")
        query_string = request.form.get("string")
        if token is None or query_string is None:
            abort(400)

        query_string = "".join(filter(is_kanji, query_string))
        if (
            not query_string.strip()
            or len(query_string) > app.config["MAX_QUERY_LENGTH"]
        ):
            abort(400)

        token = secure_filename(token)
        out_path = os.path.join(app.config["OUTPUT_PATH"], token)
        if not os.path.isdir(out_path):
            abort(404)

        return {"img_url_list": [url_for("output_file", filename="test.png")]}

    return """
    <!doctype html>
    <title>Submit new Query</title>
    <h1>Submit new Query</h1>
    <form method=post>
      token: <input type=text name=token><br>
      string: <input type=text name=string><br>
      <input type=submit value=Submit>
    </form>
    """


@app.route("/get_predefined_font_list", methods=["GET"])
def get_predefined_font_list():
    return {"token_list": ["WangXiZhi"]}


@app.route("/output/<path:filename>", methods=["GET"])
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_PATH"], filename)
