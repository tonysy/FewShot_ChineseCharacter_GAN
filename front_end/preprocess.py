# -*- coding: utf-8 -*-
import os

from config import *
import random
import requests
import PIL.Image as Image
from io import BytesIO

up_num = 7
down_num = 7
row_num = 4
column_ch_size = 135
row_ch_size = 85
fu_size = 330
upstart_L = 5
upstart_H = 130
rowstart_L = 345
rowstart_H = 15
fustart_L = 340
fustart_H = 330
color_print = (0, 0, 0)

def print_ch(background, img, L, H, ch_size):
    img = img.resize((ch_size, ch_size), Image.ANTIALIAS)
    for y in range(ch_size):
        for x in range(ch_size):
            dot = (x, y)
            color_p = img.getpixel(dot)
            if color_p < 127:
                background.putpixel((L+x, H+y), color_print)
    return background

def couplet_work(img_url_list):
    background = Image.open(couplet_pic_url)
    back_l, back_h = background.size
    L = upstart_L
    H = upstart_H
    for num in range(0, up_num):
        if IF_TEST:
            image = Image.open(img_url_list[num])
        else:
            res = requests.get(img_url_list[num])
            image = Image.open(BytesIO(res.content))
        background = print_ch(background, image, L, H, column_ch_size)
        H += column_ch_size


    L = back_l - upstart_L - column_ch_size
    H = upstart_H
    for num in range(up_num, up_num+down_num):
        if IF_TEST:
            image = Image.open(img_url_list[num])
        else:
            res = requests.get(img_url_list[num])
            image = Image.open(BytesIO(res.content))
        background = print_ch(background, image, L, H, column_ch_size)
        H += column_ch_size

    L = rowstart_L
    H = rowstart_H
    for num in range(up_num+down_num, up_num + down_num + row_num):
        if IF_TEST:
            image = Image.open(img_url_list[num])
        else:
            res = requests.get(img_url_list[num])
            image = Image.open(BytesIO(res.content))
        background = print_ch(background, image, L, H, row_ch_size)
        L += row_ch_size

    L = fustart_L
    H = fustart_H
    if IF_TEST:
        image = Image.open(img_url_list[up_num + down_num + row_num])
    else:
        res = requests.get(img_url_list[up_num + down_num + row_num])
        image = Image.open(BytesIO(res.content))
    background = print_ch(background, image, L, H, fu_size)
    
    return background
