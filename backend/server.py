import os
import imghdr
import secrets
from flask import Flask, request, abort, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from pre_process import is_kanji
from doctext import render_doc_text
from inference_api import (
    test_with_specified_chars,
    pre_defined_style_key,
    true_inferencer,
)

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
        crop_path = render_doc_text(file_path, out_path)

        crop_imgs = os.listdir(crop_path)
        crop_imgs = [item for item in crop_imgs if ".jpg" in item]
        img_paths = [os.path.join(crop_path, item) for item in crop_imgs]
        img_readed = [Image.open(item).convert("RGB") for item in img_paths]
        true_inferencer.add_new_cats(img_readed, style_key=token)

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

        if token not in pre_defined_style_key:
            pass

        token_infer = secrets.token_hex(app.config["TOKEN_LENGTH"])
        out_names = test_with_specified_chars(
            style_idx=token,
            char_list=query_string,
            direction=None,
            prefix=os.path.join(out_path, token_infer),
        )
        img_url_list = [
            url_for("output_file", filename=os.path.join(token, token_infer, out_name))
            for out_name in out_names
        ]

        return {"img_url_list": img_url_list, "kanji_string": query_string}

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)

