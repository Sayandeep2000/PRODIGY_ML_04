import cv2
import mediapipe as mp
import numpy as np
import pickle

cap = cv2.VideoCapture(0)
# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

while True:
    data_aux = []
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            # Calculate the bounding box
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Convert normalized coordinates to image coordinates
            h, w, _ = frame.shape
            min_x = int(min_x * w)
            max_x = int(max_x * w)
            min_y = int(min_y * h)
            max_y = int(max_y * h)

            # Draw the bounding box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                data_aux.append(x)
                data_aux.append(y)

        if data_aux:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            print(predicted_character)

            # Display the predicted character on the frame
            cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
