import cv2
import mediapipe as mp

# Webcam
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror camera

    # Convert BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image
    results = hands.process(imgRGB)

    h, w, c = img.shape

    # Draw X and Y axes
    cv2.line(img, (0, h // 2), (w, h // 2), (0, 255, 255), 2)  # X-axis
    cv2.line(img, (w // 2, 0), (w // 2, h), (0, 255, 255), 2)  # Y-axis

    cv2.putText(img, "X-Axis", (20, h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(img, "Y-Axis", (w // 2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Detect hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            for id, lm in enumerate(handLms.landmark):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                print(f"ID:{id}  X:{cx}  Y:{cy}")

                # Draw landmark
                cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

                # Show ID and coordinates
                cv2.putText(
                    img,
                    f"{id} ({cx},{cy})",
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 0),
                    1
                )

            # Draw hand skeleton
            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Tracking with Coordinates", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

