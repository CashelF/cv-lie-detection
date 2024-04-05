import argparse

import cv2
import mediapipe as mp
# from ffpyplayer.player import MediaPlayer

from datetime import datetime
from matplotlib import pyplot as plt
import mss
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import distance as dist

import time
import sys

from deception import process
from metrics import find_face_and_hands, is_blinking, check_hand_on_face, get_face_relative_area, hr_times, hr_values

from globals import *

TELL_MAX_TTL = 30 # how long to display a finding, optionally set in args

TEXT_HEIGHT = 30

recording = None

face_area_size = 0 # relative size of face to total frame


meter = cv2.imread('meter.png')

# BPM chart
fig = None

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', nargs='*', help='Input video device (number or path), file, or screen dimensions (x y width height), defaults to 0', default=['0'])
  parser.add_argument('--landmarks', '-l', help='Set to any value to draw face and hand landmarks')
  parser.add_argument('--bpm', '-b', help='Set to any value to draw color chart for heartbeats')
  parser.add_argument('--flip', '-f', help='Set to any value to flip resulting output (selfie view)')
  parser.add_argument('--ttl', '-t', help='How many frames for each displayed "tell" to last, defaults to 30', default='30')
  parser.add_argument('--record', '-r', help='Set to any value to save a timestamped AVI in current directory')
  args = parser.parse_args()

  if len(args.input) == 1:
    INPUT = int(args.input[0]) if args.input[0].isdigit() else args.input[0]
  elif len(args.input) != 4:
    return print("Wrong number of values for 'input' argument; should be 0, 1, or 4.")

  DRAW_LANDMARKS = args.landmarks is not None
  BPM_CHART = args.bpm is not None
  FLIP = args.flip is not None
  if args.ttl and args.ttl.isdigit():
    TELL_MAX_TTL = int(args.ttl)
  RECORD = args.record is not None

  if BPM_CHART:
    chart_setup()

  calibrated = False
  calibration_frames = 0
  with mp.solutions.face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    with mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7) as hands:
      if len(args.input) == 4:
        screen = {
          "top": int(args.input[0]),
          "left": int(args.input[1]),
          "width": int(args.input[2]),
          "height": int(args.input[3])
        }
        with mss.mss() as sct: # screenshot
          while True:
            image = np.array(sct.grab(screen))[:, :, :3] # remove alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            calibration_frames += process(image, face_mesh, hands, calibrated, DRAW_LANDMARKS, BPM_CHART)
            
            # Update chart/landmarks
            if BPM_CHART: # update chart
                fig.canvas.draw()
                fig.canvas.flush_events()

            if DRAW_LANDMARKS: # overlay face and hand landmarks
                face_landmarks, hands_landmarks = find_face_and_hands(image, face_mesh, hands)
                draw_on_frame(image, face_landmarks, hands_landmarks)

            if FLIP:
                image = cv2.flip(image, 1) # flip image horizontally

            add_text(image, tells, calibrated)
            add_truth_meter(image, len(tells))
        
            calibrated = (calibration_frames >= MAX_FRAMES)
            cv2.imshow('face', image)
            if RECORD:
              recording.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      else:
        cap = cv2.VideoCapture(INPUT)
        fps = None
        if isinstance(INPUT, str) and INPUT.find('.') > -1: # from file
          fps = cap.get(cv2.CAP_PROP_FPS)
          print("FPS:", fps)
          # cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        else: # from device
          cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
          cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
          cap.set(cv2.CAP_PROP_FPS, 30)

        if RECORD:
          RECORDING_FILENAME = str(datetime.now()).replace('.','').replace(':','') + '.avi'
          FPS_OUT = 10
          FRAME_SIZE = (int(cap.get(3)), int(cap.get(4)))
          recording = cv2.VideoWriter(
            RECORDING_FILENAME, cv2.VideoWriter_fourcc(*'MJPG'), FPS_OUT, FRAME_SIZE)

        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            calibration_frames += process(image, face_mesh, hands, calibrated, DRAW_LANDMARKS, BPM_CHART, fps)
          
            # Update chart/landmarks
            if BPM_CHART: # update chart
                fig.canvas.draw()
                fig.canvas.flush_events()

            if DRAW_LANDMARKS: # overlay face and hand landmarks
                face_landmarks, hands_landmarks = find_face_and_hands(image, face_mesh, hands)
                draw_on_frame(image, face_landmarks, hands_landmarks)

            if FLIP:
                image = cv2.flip(image, 1) # flip image horizontally
                
            calibrated = (calibration_frames >= MAX_FRAMES)
            if RECORD:
                recording.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if RECORD:
          recording.release()
  cv2.destroyAllWindows()
  
def chart_setup():
  plt.ion()
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1) # 1st 1x1 subplot
  ax.set(ylim=(185, 200))
  line, = ax.plot(hr_times, hr_values, 'b-')
  peakpts, = ax.plot([], [], 'r+')


def draw_on_frame(image, face_landmarks, hands_landmarks):
  mp.solutions.drawing_utils.draw_landmarks(
      image,
      face_landmarks,
      mp.solutions.face_mesh.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_contours_style())
  mp.solutions.drawing_utils.draw_landmarks(
      image,
      face_landmarks,
      mp.solutions.face_mesh.FACEMESH_IRISES,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_iris_connections_style())
  for hand_landmarks in (hands_landmarks or []):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style())
    
def add_text(image, tells, calibrated):
  global mood
  text_y = TEXT_HEIGHT
  if mood:
    write("Mood: {}".format(mood), image, int(.75 * image.shape[1]), TEXT_HEIGHT)
  if calibrated:
    for tell in tells.values():
      write(tell['text'], image, 10, text_y)
      text_y += TEXT_HEIGHT


def write(text, image, x, y):
  cv2.putText(img=image, text=text, org=(x, y),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 0],
    lineType=cv2.LINE_AA, thickness=4)
  cv2.putText(img=image, text=text, org=(x, y),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 255, 255],
    lineType=cv2.LINE_AA, thickness=2)
  
def add_truth_meter(image, tell_count):
  width = image.shape[1]
  sm = int(width / 64)
  bg = int(width / 3.2)

  resized_meter = cv2.resize(meter, (bg,sm), interpolation=cv2.INTER_AREA)
  image[sm:(sm+sm), bg:(bg+bg), 0:3] = resized_meter[:, :, 0:3]

  if tell_count:
    tellX = bg + int(bg/4) * (tell_count - 1) # adjust for always-on BPM
    cv2.rectangle(image, (tellX, int(.9*sm)), (tellX+int(sm/2), int(2.1*sm)), (0,0,0), 2)


if __name__ == "__main__":
    main()