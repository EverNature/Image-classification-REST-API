from datetime import datetime
from fastapi import File, UploadFile
from service.object_detection_service import get_image_object
from service.image_service import read_file_as_image, encode_image
from service.json_service import create_json
from service.image_prediction_service import image_classification

async def predict_img(
    file: UploadFile = File(...)
):
    """
    This funtion recive a file, use the object detection and predice the animal

    Inputs:
        :file: the image file

    Returns:
        ::json with all the data
    """

    predicted_classes = []
    confidences = []
    predicted = []
    msgs = []
    segmented = False

    image = read_file_as_image(await file.read())

    images, full_image, labels = get_image_object(image)

    image_pre_resized = images.copy()

    images, labels, msgs, predicted_classes, confidences, predicted, segmented = image_classification(
        images, labels, msgs, predicted_classes, confidences, predicted, segmented)

    data = create_json(image_pre_resized, segmented,
                       predicted_classes, confidences, predicted, msgs)

    im_b64 = encode_image(full_image)

    return {
        'prediction': data,
        'segmented': segmented,
        "date": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        'full_image': im_b64,
    }