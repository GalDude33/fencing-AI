import PIL.ImageOps
import numpy as np
import pyocr
import pyocr.builders
from PIL import Image


def getDigit(digitImgArr):
    img = Image.fromarray(digitImgArr)
    tools = pyocr.get_available_tools()[0]

    for i in range(6, 10 + 1):
        text = tools.image_to_string(process(img), builder=pyocr.builders.DigitBuilder(i))
        if text.isdecimal() and 0 <= int(text) and int(text) <= 15:
            break

    if not text.isdecimal():
        return getDigitByImgDiff(digitImgArr)

    return int(text)


def process(img):
    img = PIL.ImageOps.invert(img).convert('L')
    data = np.array(img)
    wb = data[...] > 20
    wb = wb.astype(np.uint8) * 255
    img = Image.fromarray(wb.astype(np.uint8))
    return img


def getDigitByImgDiff(digitImgArr):

    img = Image.fromarray(digitImgArr)
    processed_img = process(img)
    diffArr = np.zeros(16)

    for i in range(0, 16):
        curr_fixed_img = Image.open('digit_images/'+str(i)+'.png')
        curr_processed_fixed_img = process(curr_fixed_img)
        diffArr[i] = np.sum(np.asarray(processed_img)-np.asarray(curr_processed_fixed_img))/(np.sum(np.asarray(processed_img))+np.sum(np.asarray(curr_processed_fixed_img)))

    return int(np.argmin(diffArr))


# digitImgArr = np.asarray(Image.open('digit_images/15.png'))
# res=getDigitByImgDiff(digitImgArr)
# print(str(res))