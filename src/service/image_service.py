import cv2
import base64
import numpy as np

def read_file_as_image(data) -> np.ndarray:
    """
    This funtion return an array of RGB

    Inputs:
        :data: Array of bytes

    Returns:
        :: Array of RGB
    """
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return image

def encode_image(image):
    """
    Transform the array of RBG to base64

    Inputs:
        :image: the image to encode

    Returns:
        :: the image encoded
    """
    _, im_arr = cv2.imencode('.jpg', image)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)

    return im_b64