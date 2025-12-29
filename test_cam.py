import cv2
import numpy as np
import tensorflow as tf

# --------------------
# Load trained model
# --------------------
model = tf.keras.models.load_model("pain_face_transfer_best.h5")

# --------------------
# Load face detector (OpenCV DNN)
# --------------------
protoPath = r"C:\Users\rudra\Desktop\chatgpt_train3\face_detector\deploy.prototxt"
modelPath = r"C:\Users\rudra\Desktop\chatgpt_train3\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# --------------------
# Start webcam
# --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (224, 224))
            face_array = np.expand_dims(face_resized / 255.0, axis=0)

            # Prediction
            pred = model.predict(face_array, verbose=0)[0][0]

            # Confidence thresholding
            if pred > 0.7:
                label = f"Pain ({pred:.2f})"
                color = (0, 0, 255)
            else:
                label = f"No Pain ({1-pred:.2f})"
                color = (0, 255, 0)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Pain Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
