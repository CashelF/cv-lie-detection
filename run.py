import argparse

import cv2
import mediapipe as mp
import numpy as np

from config import Config
from metrics import MetricsCalculator

from video_processing import process_face_and_hands, draw_landmarks
from model import NeuralNetworkModel
from keras.models import load_model



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

def draw_meter(frame, likelihood):
    # Define colors
    red = (0, 0, 255)
    green = (0, 255, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)
    
    # Calculate meter position and size
    x, y, w, h = 70, 50, 200, 20
    
    # Calculate filled width based on likelihood
    filled_w = int(w * likelihood)
    
    # Draw outline of meter
    cv2.rectangle(frame, (x, y), (x + w, y + h), green, -1)
    
    # Draw filled portion indicating likelihood
    cv2.rectangle(frame, (x, y), (x + filled_w, y + h), red, -1)
    
    # Add labels with black background
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_margin = 5
    
    # Calculate text size
    text_size_truth = cv2.getTextSize('Truth', font, 0.5, 1)[0]
    text_size_lie = cv2.getTextSize('Lie', font, 0.5, 1)[0]
    
    # Calculate label positions
    truth_x = x - text_margin - text_size_truth[0] -1
    truth_y = y + h // 2 + text_size_truth[1] // 2
    lie_x = x + w + text_margin
    lie_y = y + h // 2 + text_size_lie[1] // 2
    
    # Draw black boxes behind the labels
    truth_box_width = text_size_truth[0] + 2 * text_margin -2
    lie_box_width = text_size_lie[0] + 2 * text_margin 
    
    # truth_box_y = y - text_size_truth[1] - text_margin
    # lie_box_y = y - text_size_lie[1] - text_margin
    truth_box_y = y +1
    lie_box_y = y +1
    
    cv2.rectangle(frame, (truth_x, truth_box_y), (truth_x + truth_box_width, truth_box_y + text_size_truth[1] + text_margin), black, -1)
    cv2.rectangle(frame, (lie_x, lie_box_y), (lie_x + lie_box_width, lie_box_y + text_size_lie[1] + text_margin), black, -1)
    
    # Draw labels
    cv2.putText(frame, 'Truth', (truth_x + text_margin, truth_y), font, 0.5, white, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Lie', (lie_x + text_margin, lie_y), font, 0.5, white, 1, cv2.LINE_AA)


def draw_metrics(frame, metrics):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_height = 20
    x, y = 20, frame.shape[0] - 20
    
    # Calculate the size of the black box
    max_metric_width = max(cv2.getTextSize(f"{metric_name}: {metric_value}", font, font_scale, 1)[0][0] for metric_name, metric_value in metrics.items())
    box_width = max_metric_width + 10
    box_height = len(metrics) * line_height + 10
    
    # Adjust the y-coordinate of the black box
    box_y = frame.shape[0] - (box_height+35)
    
    # Draw black box behind the text
    cv2.rectangle(frame, (x - 5, box_y), (x + box_width, y), (0, 0, 0), -1)
    
    # Draw metric values
    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        cv2.putText(frame, f"{metric_name}: {metric_value}", (x, y - box_height + 5 + i * line_height), font, font_scale, font_color, 1, cv2.LINE_AA)


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
  model = load_model('trained_model.h5')
  
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
        "Lip Compression": "{:.5f}".format(metrics[3]),
        "Gaze Detected": "Yes" if metrics[4] else "No",
        "Blink Rate": "{:.5f}".format(metrics[5])
      }

      # Predict using the trained model
      prediction = model.predict(input_data)
      print("Prediction:", prediction)

      draw_meter(image, prediction[0][0])
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