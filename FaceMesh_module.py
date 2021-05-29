import cv2
import mediapipe as mp
import time


# mpDraw = mp.solutions.drawing_utils
# mpFaceMesh = mp.solutions.face_mesh
# faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
# # faceMesh only accept RGB image
#
# # To change thickness of dots and lines of landmarks
# drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
#
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = faceMesh.process(imgRGB)
#     if results.multi_face_landmarks:
#         # Loop through the landmarks for all faces detected
#         for faceLms in results.multi_face_landmarks:
#             mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpecs, drawSpecs)
#
#             for lm in faceLms.landmark:
#                 # print(lm)
#                 ih, iw, ic = img.shape
#                 x, y = int(lm.x * iw), int(lm.y * ih)
#                 print(x,y)

def main():
    # For video stream
    cap = cv2.VideoCapture(0)
    pTime = 0
    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()