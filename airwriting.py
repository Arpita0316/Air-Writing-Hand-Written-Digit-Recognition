import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# ========================
# Step 1: Prepare/load MNIST model
# ========================

try:
    model = load_model("mnist_model.h5")
    print("✅ Model loaded successfully.")
except:
    print("⚠️ Model not found. Training new model...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
    model.save("mnist_model.h5")
    print("✅ Model trained and saved.")

# ========================
# Step 2: Hand Tracking Setup
# ========================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ========================
# Step 3: Drawing Canvas
# ========================

canvas = np.zeros((512,512), dtype=np.uint8)
drawing = False
prev_x, prev_y = None, None

cap = cv2.VideoCapture(0)

print("🎯 Air-writing Digit Recognition Started. Press 'q' to quit, 'c' to clear canvas, 'p' to predict digit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            x = int(handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 512)
            y = int(handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 512)
            if prev_x is None:
                prev_x, prev_y = x, y
            cv2.line(canvas, (prev_x, prev_y), (x, y), 255, 10)
            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    cv2.imshow("Air Canvas", canvas)
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((512,512), dtype=np.uint8)
    elif key == ord('p'):
        # Preprocess canvas and predict
        img = cv2.resize(canvas, (28,28))
        img = img.astype('float32') / 255.0
        img = img.reshape(1,28,28,1)
        pred = np.argmax(model.predict(img), axis=1)[0]
        print(f"🎉 Predicted Digit: {pred}")

cap.release()
cv2.destroyAllWindows()
