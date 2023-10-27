import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

# Options window
window = tk.Tk()
window.title('Chose Capture Option')
window.resizable(False, False)
app_width = 300
app_height = 100
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x_position = (screen_width / 2) - (app_width / 2)
y_position = (screen_height / 2) - (app_height / 2)
window.geometry(f'{app_width}x{app_height}+{int(x_position)}+{int(y_position)}')

# OpenCV DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn.DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


# OpenCV Tracking
def opencv_tracking(tracking_type):
    # Initialize
    capture = cv2.VideoCapture(tracking_type)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # Read frame
        ret, frame = capture.read()

        # Break out of loop if video ends
        if not ret:
            break

        # Object Detection
        all_objects_detected = []

        (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.5)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            # Draw box
            (x, y, w, h) = bbox
            get_class_name = classes[class_id]
            all_objects_detected.append(get_class_name)
            cv2.putText(frame, str(get_class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display
        objects_counts = len(all_objects_detected)
        objects_counts_text = "Object Detected: {}".format(objects_counts)
        object_counts = {}

        cv2.putText(frame, objects_counts_text, (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        for object_detected in all_objects_detected:
            if object_detected in object_counts:
                object_counts[object_detected] += 1
            else:
                object_counts[object_detected] = 1

        i = 0
        for object_name, count in object_counts.items():
            object_text = "- {}: {}".format(object_name, count)
            cv2.putText(frame, object_text, (5, 60 + i), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            i += 30

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break

    capture.release()
    cv2.destroyAllWindows()


# Open video file
def open_file():
    filepath = fd.askopenfilename(initialdir='/', title="Chose a video file", filetypes=[('video files', '*.mp4')])
    if filepath:
        showinfo(title='Selected File', message=filepath)
        opencv_tracking(filepath)


# Use camera
def use_camera():
    opencv_tracking(0)


camera_button = ttk.Button(window, text="Capture camera", command=use_camera)
camera_button.pack(expand=True)
video_button = ttk.Button(window, text="Chose Video File", command=open_file)
video_button.pack(expand=True)
window.mainloop()
