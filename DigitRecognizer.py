import PIL.ImageOps
import numpy as np
import pyocr
import pyocr.builders
from PIL import Image


def getDigit(digitImgArr):
    img = Image.fromarray(digitImgArr)

    def process(img, resize_param):
        img = img.resize(np.multiply(img.size, 2), resize_param)
        img = PIL.ImageOps.invert(img).convert('L')
        data = np.array(img)
        wb = data[...] > 20
        wb = wb.astype(np.uint8) * 255
        img = Image.fromarray(wb.astype(np.uint8))
        return img

    tools = pyocr.get_available_tools()[0]

    for i in range(6, 10 + 1):
        text = tools.image_to_string(process(img, Image.ANTIALIAS), builder=pyocr.builders.DigitBuilder(i))
        if text.isdecimal() and 0 <= int(text) and int(text) <= 15:
            break
    if not text.isdecimal():
        for i in range(6, 10 + 1):
            text = tools.image_to_string(process(img, Image.NEAREST), builder=pyocr.builders.DigitBuilder(i))
            if text.isdecimal() and 0 <= int(text) and int(text) <= 15:
                break

    return int(text)
