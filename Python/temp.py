import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
key_points = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]

def find_distances(hand_data):
    dist_matrix = np.zeros([len(hand_data), len(hand_data)], dtype='float')
    palm_size = ((hand_data[0][0] - hand_data[9][0]) ** 2 + (hand_data[0][1] - hand_data[9][1]) ** 2) ** 0.5
    for row in range(len(hand_data)):
        for col in range(len(hand_data)):
            dist_matrix[row][col] = (((hand_data[row][0] - hand_data[col][0]) ** 2 +
                                      (hand_data[row][1] - hand_data[col][1]) ** 2) ** 0.5) / palm_size
    return dist_matrix

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = image.shape
        return [(int(pt.x * w), int(pt.y * h)) for pt in hand_landmarks.landmark]
    return None

# UPDATE THIS PATH to your extracted folder:
dataset_path = r"C:\Users\abhay\Downloads\ISL_Dataset"
gest_names = []
known_gestures = []

for label in sorted(os.listdir(dataset_path)):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        image_path = os.path.join(label_path, file)
        hand_data = extract_landmarks(image_path)
        if hand_data is not None:
            dist_matrix = find_distances(hand_data)
            known_gestures.append(dist_matrix)
            gest_names.append(label)

# Save final pickle
with open("ISL_trained_full.pkl", "wb") as f:
    pickle.dump(gest_names, f)
    pickle.dump(known_gestures, f)

print(f"✅ Done! Total gestures saved: {len(gest_names)}")
