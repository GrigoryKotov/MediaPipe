import cv2
import mediapipe as mp
from roi import get_roi

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def fun():
    label = ""
    score = 0
    roi = None

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            ####### this is for testing gray image only
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            results = hands.process(image)
            # print('**************Handedness:', results.multi_handedness)
            if results.multi_handedness is not None:
                label = results.multi_handedness[0].classification[0].label
                score = results.multi_handedness[0].classification[0].score

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    if hand_landmarks.landmark[8].y > hand_landmarks.landmark[7].y or \
                            hand_landmarks.landmark[12].y > hand_landmarks.landmark[11].y or \
                            hand_landmarks.landmark[16].y > hand_landmarks.landmark[15].y or \
                            hand_landmarks.landmark[20].y > hand_landmarks.landmark[19].y:
                        print("wrong landmark")

                    else:

                        x1 = hand_landmarks.landmark[5].x
                        x2 = hand_landmarks.landmark[17].x
                        y1 = hand_landmarks.landmark[5].y
                        y2 = hand_landmarks.landmark[17].y
                        roi = get_roi(image, x1, x2, y1, y2, label)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            if roi is not None:
                cv2.imshow("Palm ROI", roi)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cv2.waitKey(0)
    cap.release()

fun()