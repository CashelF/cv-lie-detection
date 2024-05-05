import argparse

import cv2
import mediapipe as mp
import numpy as np

from config import Config
from metrics import MetricsCalculator

from video_processing import process_face_and_hands, draw_landmarks, draw_meter, draw_metrics
from model import NeuralNetworkModel
import tensorflow as tf

import pickle

def process_input_data(running_delta_BPM, running_delta_emotion, running_delta_hands, running_delta_lip, running_delta_gaze, running_delta_blink):
    # Convert delta arrays to numpy arrays
    input_data = np.array([
        running_delta_BPM[-1],
        running_delta_emotion[-1],
        running_delta_hands[-1],
        running_delta_lip[-1],
        running_delta_gaze[-1],
        running_delta_blink[-1]
    ])
    # Reshape the input data to match the model's input shape
    input_data = input_data.reshape(1, -1)  # Assuming the model expects a 2D array with one row
    return input_data

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
  model_filename = 'finalized_xgb_classifier.pkl'
  model = pickle.load(open(model_filename, 'rb'))
  
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
      #print("Metrics: BPM, emotion, hands, lip, gaze, blink", metrics)
      

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
        running_avg_BPM = metrics[0] if metrics[0] != None else 0
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

      input_data = process_input_data(running_delta_BPM, running_delta_emotion, running_delta_hands, running_delta_lip, running_delta_gaze, running_delta_blink)


      emotions = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral"
      }
    
      metrics_dict = {
        "BPM": metrics[0],
        "Emotion": emotions.get(metrics[1]),
        "Hands Detected": "Yes" if metrics[2] else "No",
        "Lip Compression": metrics[3],
        "Gaze Detected": "Yes" if metrics[4] else "No",
        "Blink Rate": metrics[5]
      }

      # Predict using the trained model
      prediction = model.predict_proba(input_data)
      print("Prediction:", prediction[0][1])

      draw_meter(image, prediction[0][1])
      draw_metrics(image, metrics_dict)

      counter+=1
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