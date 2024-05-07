import argparse

import cv2
import mediapipe as mp
import pandas as pd

from config import Config
from metrics import MetricsCalculator

from image_processing import process_face_and_hands, draw_landmarks, draw_metrics, draw_meter

from joblib import load

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', nargs='*', default=['0'], help='Input video device (number or path), file, or screen dimensions (x y width height), defaults to 0')

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
  
  model = load('model.joblib')
  
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
      metrics_dict = {
        "BPM": [metrics[0] or 80], # default 80
        "Emotion": [metrics[1]],
        "Hands": [metrics[2]],
        "Lip": [metrics[3]],
        "Gaze": [metrics[4]],
        "Blink": [metrics[5]]
      }

      # Convert the dictionary to a DataFrame
      metrics_df = pd.DataFrame(metrics_dict)

      # Predict using the trained model
      prediction = model.predict_proba(metrics_df)[0][0]

      draw_meter(image, prediction)
      draw_metrics(image, metrics)

      draw_landmarks(image, face_landmarks, hands_landmarks)
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