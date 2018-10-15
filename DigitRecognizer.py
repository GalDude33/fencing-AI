import PIL.ImageOps
import numpy as np
import pyocr
import pyocr.builders
from PIL import Image


def getDigit(digitImgArr):
    img = Image.fromarray(digitImgArr)
    img = img.resize((60, 80), Image.ANTIALIAS)
    img = PIL.ImageOps.invert(img).convert('L')
    data = np.array(img)
    wb = data[...] > 20
    wb = wb.astype(np.uint8) * 255
    img = Image.fromarray(wb.astype(np.uint8))
    tools = pyocr.get_available_tools()[0]
    text = tools.image_to_string(img, builder=pyocr.builders.DigitBuilder(7))
    return int(text)

