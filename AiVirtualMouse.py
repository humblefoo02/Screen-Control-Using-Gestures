import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui
import matplotlib.pyplot as plt

##########################
wCam, hCam = 640, 480
frameR = 90  # frame reduction
smoothening = 2
fps = 60
##########################

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # this is the magic!

cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
cap.set(cv2.CAP_PROP_FPS, fps)

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

start_time = time.time()
frame_count = 0

detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()

commands_executed = {
    'Move Mouse': 0,
    'Left Click': 0,
    'Right Click': 0,
    'Scroll Up': 0,
    'Scroll Down': 0,
    # 'Take Screenshot': 0,
    'Minimize Window': 0
}

total_frames = 0
successful_detections = 0

# Keep the camera window on top
cv2.namedWindow("image", cv2.WND_PROP_TOPMOST)
cv2.setWindowProperty("image", cv2.WND_PROP_TOPMOST, 1)

while True:

    # 1 Find hand positions
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Check if landmarks were successfully detected
    if lmList:
        successful_detections += 1

    total_frames += 1

    # 2 Tip of index and middle fingers [8, 12]
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        xt, yt = lmList[4][1:]
        # print(x1, y1, x2, y2)

        # 3 Check the fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (225, 165, 0), 2)

        # 4 Only index finger: Moving
        if fingers[1] == 1 and fingers[2] == 0:

            commands_executed['Move Mouse'] += 1 / fps

            # 5 Convert the coordinates
            if frameR < x1 < wCam - frameR and frameR < y1 < hCam - frameR:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # 6 Smoothen the values
                clocX = int(plocX + (x3 - plocX) / smoothening)
                clocY = int(plocY + (y3 - plocY) / smoothening)

                # 7 Move our mouse
                pyautogui.moveTo(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255),
                           cv2.FILLED)
                plocX, plocY = clocX, clocY



        # 8 Check the clicking condition (left click)
        if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:

            commands_executed['Left Click'] += 1 / fps

            # 9 Find the distance among the fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)

            # 10 click mouse if distance are as per our need
            if length < 35:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (255, 255, 0),
                           cv2.FILLED)
                pyautogui.click()
                time.sleep(0.1)




        # right click functionality
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:

            commands_executed['Right Click'] += 1 / fps

            # 9 Find the distance among the fingers
            length, img, lineInfo = detector.findDistance(4, 8, img)
            # print(length)

            # 10 click mouse if distance are as per our need
            if length < 50:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (255, 255, 0), cv2.FILLED)
                pyautogui.click(button='right')  # Perform right-click
                time.sleep(0.1)




        # scrolling functionality
        if fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[4] == 1:
            cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
            if fingers[3] == 1:
                pyautogui.scroll(-50)
                commands_executed['Scroll Up'] += 1 / fps
            else:
                pyautogui.scroll(50)
                commands_executed['Scroll Down'] += 1 / fps




        # minimize window
        if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 1:
            pyautogui.hotkey('win', 'down')  # Windows: Win + Down arrow
            commands_executed['Minimize Window'] += 1 / fps
            # pyautogui.hotkey('command', 'm')  # macOS: Command + M (this might vary)
            # pyautogui.hotkey('ctrl', 'm')  # Linux: Ctrl + M (this might vary)
            time.sleep(1)


        frame_count += 1

        # # Check if landmarks were successfully detected
        # if lmList:
        #     successful_detections += 1

        # break
        if fingers[2] == 1 and fingers[3] == 1 and fingers[0] == 0 and fingers[1] == 0 and fingers[4] == 0:
            break


    # 11 frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0),
                3)

    # 12 display
    cv2.imshow("image", img)
    cv2.waitKey(1)



# Calculate total execution time
total_time = time.time() - start_time

# Visualize commands executed
commands = list(commands_executed.keys())
times = list(commands_executed.values())

plt.figure(figsize=(10, 6))
bars = plt.barh(commands, times, color='skyblue')
plt.xlabel('Time (seconds)')
plt.ylabel('Commands')
plt.title('Duration of Commands Execution')
plt.grid(axis='x')

# Add total time below the graph
plt.text(0, -0.5, f'Total Execution Time: {total_time:.2f} seconds', fontsize=10)

# Add times on bars
for bar, time in zip(bars, times):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{time:.2f}s',
             ha='left', va='center', color='black')

plt.show()

# Calculate accuracy percentage
accuracy_percentage = (successful_detections / total_frames) * 100 if total_frames > 0 else 0

# Calculate average FPS
average_fps = frame_count / total_time if total_time > 0 else 0

# Print accuracy percentage and average FPS
print(f'Accuracy: {accuracy_percentage:.2f}%')
print(f'Average FPS: {average_fps:.2f}')

# Add accuracy and average FPS below the graph
plt.text(0, -0.7, f'Accuracy: {accuracy_percentage:.2f}%', fontsize=10)
plt.text(0, -0.9, f'Average FPS: {average_fps:.2f}', fontsize=10)

# Visualize accuracy
plt.figure(figsize=(6, 6))
labels = ['Correct Detections', 'Incorrect Detections']
sizes = [successful_detections, total_frames - successful_detections]
colors = ['lightgreen', 'lightcoral']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Hand Detection Accuracy')

plt.show()