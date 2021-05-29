import cv2
import mediapipe as mp
import time

class FaceMeshModule():
    def __init__(self, staticMode=False, maxFaces=2,minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetectionCon=minDetectionCon
        self.minTrackCon=minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces, self.minDetectionCon, self.minTrackCon)
        # faceMesh only accept RGB image

        # To change thickness of dots and lines of landmarks
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFacelms(self,img,draw=True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            # Loop through the landmarks for all faces detected
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpecs, self.drawSpecs)

                face=[]
                # To store face landmarks in list
                for id,lm in faceLms.landmark:
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    print(id,x,y)
                    face.append([x,y])
            faces.append(face)
        return img, faces

def main():
    # For video stream
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = FaceMeshModule()

    while True:
        success, img = cap.read()
        img=detector.findFacelms(img,True)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()