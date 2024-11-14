import matplotlib.pyplot as plt
import cv2
import logging


def set_logging(show_logs=True):
    if show_logs:
        logging.getLogger("ultralytics").setLevel(logging.INFO)
    else:
        logging.getLogger("ultralytics").setLevel(logging.ERROR)


def model_predict(model, img_path, show_pred=False, logs=False):
    if not logs:
        set_logging(show_logs=False)

    result = model(img_path)

    if show_pred:
        annotated_img = result[0].plot()
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    if not logs:
        set_logging(show_logs=True)

    return result
