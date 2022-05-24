from service.image_service import encode_image

def create_json(images, segmented, predicted_classes, confidences, predicted, msgs):
    """
    This function creates a json

    Inputs:
        :images: an array of all the images that opencv segmented
        :segmented: the image has been segmented
        :predicted_classes: an array of all the predicted classes
        :confidences: an array of all the conficendes
        :predicted: the model has been used
        :msg: the message that is gonna be send

    Returns:
        :: a json
    """

    data = []
    for i in range(len(images)):
        if segmented:
            encode_image(images[i])
            item1 = {"clasc": predicted_classes[i], "confidence": str(confidences[i]), "predicted": predicted[i], "msg": msgs[i], "image": encode_image(images[i])}
            data.append(item1)
        else:
            item2 = {"clasc": "", "confidence": "", "predicted": "", "msg": "", "image": ''}
            data.append(item2)
    return data