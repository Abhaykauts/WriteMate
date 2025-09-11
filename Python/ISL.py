import time
import cv2
import mediapipe as mp
import numpy as np
import pickle


class mpHands:
    def __init__(self, maxHands=2, tol1=0.5, tol2=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=tol1,
            min_tracking_confidence=tol2
        )

    def Marks(self, frame):
        myHands = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        if results.multi_hand_landmarks:
            for handLandMarks in results.multi_hand_landmarks:
                myHand = []
                for landMark in handLandMarks.landmark:
                    myHand.append((int(landMark.x * width), int(landMark.y * height)))
                myHands.append(myHand)
        return myHands


def findDistances(handData):
    distMatrix = np.zeros([len(handData), len(handData)], dtype='float')
    palmSize = ((handData[0][0] - handData[9][0]) ** 2 + (handData[0][1] - handData[9][1]) ** 2) ** 0.5
    for row in range(len(handData)):
        for column in range(len(handData)):
            distMatrix[row][column] = (((handData[row][0] - handData[column][0]) ** 2 +
                                        (handData[row][1] - handData[column][1]) ** 2) ** 0.5) / palmSize
    return distMatrix


def findError(gestureMatrix, unknownMatrix, keyPoints):
    error = 0
    for row in keyPoints:
        for column in keyPoints:
            error += abs(gestureMatrix[row][column] - unknownMatrix[row][column])
    return error


def findGesture(unknownGesture, knownGestures, keyPoints, gestNames, tol):
    errorArray = []
    for i in range(len(gestNames)):
        error = findError(knownGestures[i], unknownGesture, keyPoints)
        errorArray.append(error)
    minError = min(errorArray)
    if minError < tol:
        gesture = gestNames[errorArray.index(minError)]
    else:
        gesture = 'Unknown'
    return gesture


# Setup camera
width = 720
height = 360
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Initialize the hand detector
findHands = mpHands(maxHands=2)
time.sleep(2)  # Allow camera to initialize

keyPoints = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]

train = int(input('Enter 1 to Train, Enter 0 to Recognize: '))

if train == 1:
    trainCnt = 0
    knownGestures = []
    numGest = 36  # 26 letters + 10 digits
    gestNames = []
    for i in range(numGest):
        if i < 26:
            name = chr(65 + i)  # A-Z for alphabets    A=65
        else:
            name = str(i - 26)  # 0-9 for digits
        gestNames.append(name)

    trainName = input('Filename for training data? (Press Enter for Default): ')
    if trainName == '':
        trainName = 'default'
    trainName += '.pkl'

else:
    trainName = input('What Training Data Do You Want to Use? (Press Enter for Default): ')
    if trainName == '':
        trainName = 'default'
    trainName += '.pkl'
    try:
        with open(trainName, 'rb') as f:
            gestNames = pickle.load(f)
            knownGestures = pickle.load(f)
    except FileNotFoundError:
        print("Training data file not found. Exiting.")
        exit()

tol = 10
gesture_string = ""
last_gesture_time = 0  # To track the last gesture recognition time
last_append_time = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (width, height)) #double check
    handData = findHands.Marks(frame)

    if train == 1 and handData:
        print(f'Please Show Gesture {gestNames[trainCnt]}: Press t when Ready')
        if cv2.waitKey(1) & 0xFF == ord('t'):
            knownGesture = findDistances(handData[0])
            knownGestures.append(knownGesture)
            trainCnt += 1
            if trainCnt == numGest:
                train = 0
                with open(trainName, 'wb') as f:
                    pickle.dump(gestNames, f)
                    pickle.dump(knownGestures, f)
                print("Training complete. Data saved.")
    elif train == 0 and handData:
        current_time = time.time()
        if current_time - last_gesture_time > 0.8:
            unknownGesture = findDistances(handData[0])
            myGesture = findGesture(unknownGesture, knownGestures, keyPoints, gestNames, tol)

            if myGesture != 'Unknown':
                if current_time - last_append_time >= 2:
                    gesture_string += ' '  # Add space between words
                gesture_string += myGesture
                last_append_time = current_time  # Update when gesture was appended
                cv2.putText(frame, myGesture, (100, 175), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 8)

            last_gesture_time = current_time  # Update when a gesture was processed

            # Update the last gesture time
            print(f'Current Gesture String: {gesture_string}')

    for hand in handData:
        for ind in keyPoints:
            cv2.circle(frame, hand[ind], 25, (255, 0, 255), 3)  # Show tracking points

    cv2.imshow('my WEBcam', frame)
    # cv2.moveWindow('my WEBcam', 0, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Convert gesture string to SVG at the end
import svgwrite

dwg = svgwrite.Drawing('gesture_output.svg', profile='tiny')
dwg.add(dwg.text(gesture_string, insert=(10, 20), fill='black', font_size="20px"))
dwg.save()

cam.release()
cv2.destroyAllWindows()