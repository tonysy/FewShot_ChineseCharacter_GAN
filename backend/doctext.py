import argparse
from enum import Enum
import io
import os

from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
import cv2

from pre_process import binarize, is_kanji


def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon(
            [
                bound.vertices[0].x,
                bound.vertices[0].y,
                bound.vertices[1].x,
                bound.vertices[1].y,
                bound.vertices[2].x,
                bound.vertices[2].y,
                bound.vertices[3].x,
                bound.vertices[3].y,
            ],
            None,
            color,
        )
    return image


def get_document_bounds(image_file):
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient()

    bounds = []
    text = []

    with io.open(image_file, "rb") as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        bounds.append(symbol.bounding_box)
                        text.append(symbol.text)

    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds, text


def render_doc_text(file_in, path_out):
    bounds, text = get_document_bounds(file_in)

    image = Image.open(file_in)
    draw_boxes(image, bounds, "red")
    image.save(os.path.join(path_out, "rendered.jpg"))

    img = cv2.imread(file_in)
    crop_path = os.path.join(path_out, "crop")
    os.mkdir(crop_path)
    for bound, t in zip(bounds, text):
        if is_kanji(t):
            min_x = min(v.x for v in bound.vertices)
            min_y = min(v.y for v in bound.vertices)
            max_x = max(v.x for v in bound.vertices)
            max_y = max(v.y for v in bound.vertices)
            crop = img[min_y:max_y, min_x:max_x]
            crop = binarize(crop)
            cv2.imwrite(os.path.join(crop_path, "{}.jpg".format(t)), crop)

    return crop_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("detect_file", help="The image for text detection.")
    parser.add_argument("out_path", help="Output path")
    args = parser.parse_args()

    render_doc_text(args.detect_file, args.out_path)
