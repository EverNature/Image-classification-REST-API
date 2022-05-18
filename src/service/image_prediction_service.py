import tensorflow as tf
import numpy as np

MODEL = tf.keras.models.load_model(
     "C:/Users/aritz/OneDrive/Escritorio/MUUUH/3. maila/2/PBL/IA/saved_models/vgg16_weights_full_model_v2_standarization")

LABEL_NAMES = ['bird', 'cat', 'dog', 'horse', 'sheep',
     'cow', 'elephant', 'bear', 'zebra', 'giraffe']

CLASS_NAMES = ['ABBOTTS BABBLER',
               'ABBOTTS BOOBY',
               'AFRICAN FIREFINCH',
               'AFRICAN OYSTER CATCHER',
               'BEARDED BARBET',
               'CASPIAN TERN',
               'CEDAR WAXWING',
               'GREAT KISKADEE',
               'HOUSE FINCH']


def image_classification(images, labels, msgs, predicted_classes, confidences, predicted, segmented):
    for i in range(len(images)):

        if labels[i] == 'bird':
            images[i] = tf.image.resize(images[i], [224, 224])
            img_batch = np.expand_dims(images[i], 0)

            predictions = MODEL.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            if confidence < 0.50:
                predicted_class = ''
                confidence = 0.0
                msgs.append('Unrecognized animal predicted by the model')
            else:
                msgs.append('Recognized animal')

            predicted_classes.append(predicted_class)
            confidences.append(confidence)
            predicted.append(True)
            segmented = True

        elif labels[i] in LABEL_NAMES:
            predicted_class = ''
            predicted_classes.append(predicted_class)
            confidence = 0.0
            confidences.append(confidence)
            predicted.append(True)
            msgs.append('Unknown animal')
            segmented = True

        else:
            predicted_class = ''
            predicted_classes.append(predicted_class)
            confidence = 0.0
            confidences.append(confidence)
            predicted.append(False)
            msgs.append('No animal detected')

    return images, labels, msgs, predicted_classes, confidences, predicted, segmented
