import skimage
import skimage.io
import skimage.transform
import numpy as np

from skimage.color import rgba2rgb

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = rgba2rgb(img)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, names):
    targets = [n.split('_')[3] == '1.png' and '1' or '0' for n in names]
    for i in range(len(targets)):
        pred = np.argmax(prob[i])
        print(targets[i], pred, prob[i][pred])
