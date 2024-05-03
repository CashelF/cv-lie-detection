import argparse

import cv2
import mediapipe as mp
import os
import csv

from config import Config
from metrics import MetricsCalculator

from video_processing import process_face_and_hands, draw_landmarks

def main():
  # parser = argparse.ArgumentParser()
  # parser.add_argument('--input', '-i', nargs='*', default=['0'], help='Input video device (number or path), file, or screen dimensions (x y width height), defaults to 0')
  # parser.add_argument('--mesh', '-m', action='store_true', help='Enable drawing of face and hand mesh')

  # args = parser.parse_args()

  # if len(args.input) == 1 and args.input[0].isdigit():
  #     INPUT = int(args.input[0])
  # else:
  #     INPUT = args.input[0]

  directory = r'C:\Users\jibra\Desktop\ECE379K\cvproject\Real-life_Deception_Detection_2016\Clips\Deceptive'
  files = os.listdir(directory)
  with open(r'C:\Users\jibra\Desktop\ECE379K\cvproject\cv-lie-detection\all_videos_metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    #writer.writerow(['Video', 'Delta BPM', 'Delta Emotion', 'Delta Hands', 'Delta Lip', 'Delta Gaze', 'Delta Blink', 'Deception'])
    for file in files:
      INPUT = os.path.join(directory, file)
      cap = cv2.VideoCapture(INPUT)
      Config.FPS = cap.get(cv2.CAP_PROP_FPS) # time tell shows should be 1 second
      if isinstance(INPUT, str) and INPUT.find('.') == -1: # from camera, not file
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)


      face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
      hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
      running_avg_BPM = 0
      running_avg_lip = 0
      running_avg_blink = 0
      running_avg_emotion = 0
      running_avg_hands = 0
      running_avg_gaze = 0

      running_delta_BPM = []
      running_delta_emotion = []
      running_delta_hands = []
      running_delta_lip = []
      running_delta_gaze = []
      running_delta_blink = []

      counter = 0
      gaze_flag = 0
      
      try:

        metrics_calculator = MetricsCalculator()
        while cap.isOpened():
          success, image = cap.read()
          if not success: 
            break

          face_landmarks, hands_landmarks = process_face_and_hands(image, face_mesh, hands)
          if not face_landmarks: 
            continue
          
          metrics = metrics_calculator.collect_metrics(image, face_landmarks, hands_landmarks)
          print("Metrics: BPM, emotion, hands, lip, gaze, blink", metrics)
          

          #in a separate array, store the running average for each frame. This essentially just takes the average of the previous values
          #and adds the current one and averages it out 
          #do it 





          if(metrics[4] == True): #appends 1 to the delta array if gaze is detected
            running_avg_gaze = 1
            running_delta_gaze.append(1)
            gaze_flag = 30
          elif gaze_flag > 0: #appends 1 to the delta array for 30 frames after gaze is detected
            running_delta_gaze.append(1- running_avg_gaze)
            running_avg_gaze = (1+running_avg_gaze)/(counter+1)
            gaze_flag -= 1
          else: #append 0 if gaze is not detected
            running_delta_gaze.append(0-running_avg_gaze)
            running_avg_gaze = (0+running_avg_gaze)/(counter+1)
          

          if(counter==0):
            if(metrics[0] == None):
              running_avg_BPM = 0
            else:
              running_avg_BPM = metrics[0]
            running_delta_BPM.append(0)
            running_avg_emotion = metrics[1]
            running_delta_emotion.append(0)
            if(metrics[2] == True):
              running_avg_hands = 1
              running_delta_hands.append(1) #append 1 if hands are detected
            else:
              running_avg_hands = 0
              running_delta_hands.append(0)
            running_avg_lip = metrics[3]
            running_delta_lip.append(0)
            running_avg_blink = metrics[5]
            running_delta_blink.append(0)

          else:
            if(metrics[0] == None):
              running_delta_BPM.append(0) 
            else:
              running_delta_BPM.append(metrics[0]-running_avg_BPM)
              running_avg_BPM = (metrics[0]+running_avg_BPM)/(counter+1)
            running_delta_emotion.append(metrics[1]-running_avg_emotion)
            running_avg_emotion = (metrics[1]+running_avg_emotion)/(counter+1)
            if(metrics[2] == True):
              running_delta_hands.append(1-running_avg_hands)
              running_avg_hands = (1+running_avg_hands)/(counter+1)
            else:
              running_delta_hands.append(0-running_avg_hands)
              running_avg_hands = (0+running_avg_hands)/(counter+1)
            running_delta_lip.append(metrics[3]-running_avg_lip)
            running_avg_lip = (metrics[3]+ running_avg_lip)/(counter+1)
            running_delta_blink.append(metrics[5]-running_avg_blink)
            running_avg_blink = (metrics[5]+running_avg_blink)/(counter+1)

          counter+=1
          writer.writerow([file, running_delta_BPM[-1], running_delta_emotion[-1], running_delta_hands[-1], running_delta_lip[-1], running_delta_gaze[-1], running_delta_blink[-1], 1])

          # if args.mesh: draw_landmarks(image, face_landmarks, hands_landmarks)
          cv2.imshow('face', image)
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
          
      finally:
        cap.release()
        face_mesh.close()
        hands.close()
        cv2.destroyAllWindows()
  

if __name__ == '__main__':
  main()