from ultralytics import YOLO
from matplotlib import pyplot as plt

model1 = YOLO("C:/Users/Public/pycharm/Yolo_detection/car_detection_model/best.pt")
model2 = YOLO("C:/Users/Public/pycharm/Yolo_detection/car_detection_model/bests.pt")
def detect_image(image_path,model):
    # Предсказание
    results = model.predict(image_path, conf=0.5, save=True)

    # Визуализация с Matplotlib
    for r in results:
        im_array = r.plot()  # Рисуем bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(im_array[..., ::-1])  # BGR -> RGB
        plt.axis('off')
        plt.show()

detect_image("C:/Users/igork/Downloads/2bb37f15ebbfe66e68bd0b02d5061b57.jpg",model1)
detect_image("C:/Users/igork/Downloads/2bb37f15ebbfe66e68bd0b02d5061b57.jpg",model2)