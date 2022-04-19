import cv2
import mediapipe as mp
from time import time

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture('KneeBendVideo.mp4')

counter = 0
stage = 'relax'
timer = 0
side = 'left'
showError = False
timeofError = int()
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        # image = cv2.flip(image,1)
        if not success:
            print("Video Finished or Empty!")
            break
        image.flags.writeable = False
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


        #extracting landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            if landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z > landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z :
                if side == 'left':
                    counter=0
                side = 'right'
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                angle = calculate_angle(hip,knee,ankle)
                
                cv2.putText(image, str(int(angle)),
                    tuple(np.multiply(knee,[854,640]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255),
                )
                angle = int(angle)
                
                if showError and time()-timeofError <= 3:
                    cv2.rectangle(image,(255,50),(630,123),(0,0,0),-1)
                    cv2.putText(image, 'Keep your knee bent',
                    (275,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255),
                    thickness=1
                )
                if time()-timeofError > 5:
                    showError = False
                if angle < 140 and stage == 'relax':
                    timer=time()
                    stage='contract'
                
                elif angle > 140 and stage == 'contract' and time()-timer>=8:
                    stage = 'relax'
                    counter+=1
                    print(counter)


                elif angle > 140 and stage == 'contract' and time()-timer<8:
                    print('Keep your knee bent')
                    stage = 'relax'
                    print(int(timer-time()))
                    showError = True
                    timeofError = time()
                cv2.rectangle(image,(0,0),(225,73),(150,117,16),-1)
                cv2.putText(image, 'REPS:',
                    (15,32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255),
                    thickness=1
                )
                cv2.putText(image, str(counter),
                    (15,62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0),
                )
            else:
                if side == 'right':
                    counter = 0
                side = 'left'
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(hip,knee,ankle)

                cv2.putText(image, str(int(angle)),
                    tuple(np.multiply(knee,[854,640]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255),
                )
                angle = int(angle)

                if showError and time()-timeofError <= 3:
                    cv2.rectangle(image,(255,50),(630,123),(0,0,0),-1)
                    cv2.putText(image, 'Keep your knee bent',
                    (275,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255),
                    thickness=1
                )
                if time()-timeofError > 5:
                    showError = False
                if angle < 140 and stage == 'relax':
                    timer=time()
                    stage='contract'
                

                elif angle > 140 and stage == 'contract' and time()-timer>=8:
                    stage = 'relax'
                    counter+=1
                    print(counter)
                elif angle > 140 and stage == 'contract' and time()-timer<8:
                    print('Keep your knee bent')
                    stage = 'relax'
                    print(int(timer-time()))
                    showError = True
                    timeofError = time()

            cv2.rectangle(image,(0,0),(225,73),(150,117,16),-1)
            cv2.putText(image, 'REPS:',
                    (15,32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255),
                    thickness=1
                )
            cv2.putText(image, str(counter),
                    (15,62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0),
                )
            

        except:
            pass


        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),#colors of our dots
            mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)#colors of our connection lines
        )

        cv2.imshow('Knee Exercise',image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
