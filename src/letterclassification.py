from os import listdir
from os.path import isfile, join
from skimage.viewer import ImageViewer
import numpy as np
from PIL import Image

basedir = "/Users/casey/Downloads/handwrittenmathsymbols/extracted_images-1/"


def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 1))
    return im_arr


def letterdir(letter, return_array=False, return_path=False, limit=None):
    try:
        letter = letter.upper()
        onlyfiles = [
            f'{basedir+letter+"/"+f}'
            for f in listdir(basedir + letter)
            if isfile(join(basedir + letter, f))
        ]
    except:
        letter = letter.lower()
        onlyfiles = [
            f'{basedir+letter+"/"+f}'
            for f in listdir(basedir + letter)
            if isfile(join(basedir + letter, f))
        ]

    if return_array:
        onlyfiles = np.random.choice(onlyfiles, size=limit, replace=False)
        return [np.array(Image.open(X)) for X in onlyfiles]
    if return_path:
        if limit:
            onlyfiles = np.random.choice(onlyfiles, size=limit, replace=False)
            return onlyfiles
        else:
            return onlyfiles


def word2img(word, show=True, ):
    left_edge_blank = np.ones((45, 1)) * 0
    for letter in word[::-1]:
        if letter == " ":
            img2ary = np.ones((45, 45)) * np.max(left_edge_blank)
        else:
            img2ary = letterdir(letter, return_array=True, limit=1)[0]

        left_edge_blank = np.hstack((img2ary, left_edge_blank))
    if show:
        Image.fromarray(left_edge_blank).show()
        return left_edge_blank[:, 1::]
    else:
        return left_edge_blank[:, 1::]


sentance1 = "handwritten characters from mathematical expressions"
sentance1 = "including arabic and greek characters "
sentance1 = "handwritten characters from mathematical expressions"

output = word2img(long_paragraph, show=True)
# output_reshaped = output.reshape((,200*45))
