import PIL.ImageOps
import numpy as np
import pyocr
import pyocr.builders
from PIL import Image
from scipy.signal import correlate2d


def process(img):
    img = PIL.ImageOps.invert(img).convert('L')
    data = np.array(img)
    wb = data[...] > 20
    wb = wb.astype(np.uint8) * 255
    img = Image.fromarray(wb.astype(np.uint8))
    return img


static_digits_map = {}

for i in range(16):
    curr_fixed_img = Image.open('digit_images/' + str(i) + '.png')
    curr_processed_fixed_img = process(curr_fixed_img)
    static_digits_map[str(i)] = curr_processed_fixed_img


def getDigit(digitImgArr):
    # img = Image.fromarray(digitImgArr)
    # tools = pyocr.get_available_tools()[0]
    #
    # for i in range(6, 10 + 1):
    #     text = tools.image_to_string(process(img), builder=pyocr.builders.DigitBuilder(i))
    #     if text.isdecimal() and 0 <= int(text) and int(text) <= 15:
    #         break
    #
    # if not text.isdecimal():
    #     return getDigitByImgDiff(digitImgArr)
    #
    # return int(text)
    return getDigitByImgDiff(digitImgArr)


def getDigitByImgDiff(digitImgArr):
    img = Image.fromarray(digitImgArr)
    processed_img = process(img)
    processed_img = np.pad(processed_img, [2,2], mode='edge')

    digit = np.argmax([np.max(correlate2d(np.subtract(static_digits_map[str(i)], np.mean(static_digits_map[str(i)])),
                               np.subtract(processed_img, np.mean(processed_img)),
                               boundary='fill',
                               mode='valid',
                               fillvalue=255)) for i in range(16)])
    return digit
