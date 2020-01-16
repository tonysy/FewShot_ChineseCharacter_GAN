import os
import cv2


def is_kanji(c):
    return c >= u"\u4e00" and c <= u"\u9fa5"


def binarize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


if __name__ == "__main__":
    input_path = "Calligraphy"
    output_path = "Calligraphy_Processed"
    text_file = "Text"
    CUT_H = 60

    os.mkdir(output_path)
    f = open(os.path.join(input_path, text_file), "r", encoding="utf8")
    labels = "".join(filter(is_kanji, f.read()))

    for filename in os.listdir(input_path):
        if filename.split(".")[-1] == "jpg":
            img = cv2.imread(os.path.join(input_path, filename))

            h, w, _ = img.shape
            cut_h = (h - w) // 2
            if cut_h >= CUT_H:
                img = img[cut_h:-cut_h, :]
            else:
                print(filename, cut_h)
                continue

            img = binarize(img)
            label = labels[int(filename.split(".")[0]) - 1]
            cv2.imwrite(os.path.join(output_path, label + ".jpg"), img)
