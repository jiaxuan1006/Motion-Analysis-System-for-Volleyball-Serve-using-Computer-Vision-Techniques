import cv2
import numpy as np
import time

class ObjectDetection:
    def __init__(self, weights_path="c:/Users/User/Documents/source_code/dnn_model/yolov4.weights", cfg_path="c:/Users/User/Documents/source_code/dnn_model/yolov4.cfg"):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 416  # Reducing size for faster processing

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="c:/Users/User/Documents/source_code/dnn_model/classes.txt"):
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)
        return self.classes

    def detect(self, frame):
        start_time = time.time()
        classes, confidences, boxes = self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
        end_time = time.time()

        print(f"Detection time: {end_time - start_time:.4f} seconds")
        return classes, confidences, boxes

# Example usage
if __name__ == "__main__":
    detector = ObjectDetection()

    # Load a sample image
    image_path = "path/to/your/image.jpg"
    frame = cv2.imread(image_path)

    # Perform detection
    classes, confidences, boxes = detector.detect(frame)

    # Draw detection results
    for (classid, confidence, box) in zip(classes, confidences, boxes):
        color = detector.colors[int(classid) % len(detector.colors)]
        label = f"{detector.classes[classid]}: {confidence:.2f}"
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image
    cv2.imshow("Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()