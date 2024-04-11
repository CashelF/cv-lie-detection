import argparse

import cv2
import mediapipe as mp

from config import Config
from metrics import MetricsCalculator

from video_processing import process_face_and_hands, draw_landmarks

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', nargs='*', default=['0'], help='Input video device (number or path), file, or screen dimensions (x y width height), defaults to 0')
  parser.add_argument('--mesh', '-m', action='store_true', help='Enable drawing of face and hand mesh')

  args = parser.parse_args()

  if len(args.input) == 1 and args.input[0].isdigit():
      INPUT = int(args.input[0])
  else:
      INPUT = args.input[0]

  
  cap = cv2.VideoCapture(INPUT)
  Config.FPS = cap.get(cv2.CAP_PROP_FPS) # time tell shows should be 1 second
  if isinstance(INPUT, str) and INPUT.find('.') == -1: # from camera, not file
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)


  face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
  hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
  running_avg_BPM = []
  running_avg_emotion = []
  running_avg_hands = []
  running_avg_lip = []
  running_avg_gaze = []
  running_avg_blink = []
  counter = 0
  
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
      if(counter==0):
        running_avg_lip.append(metrics[3])
        running_avg_gaze.append(metrics[4])
        running_avg_blink.append(metrics[5])

      else:
        running_avg_lip.append((metrics[3]+running_avg_lip[counter-1])/(counter+1))
        running_avg_gaze.append((metrics[4]+running_avg_gaze[counter-1])/(counter+1))
        running_avg_blink.append((metrics[5]+running_avg_blink[counter-1])/(counter+1))

      counter+=1
      
      if args.mesh: draw_landmarks(image, face_landmarks, hands_landmarks)
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