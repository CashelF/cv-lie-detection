# deception.py
import numpy as np

# Constants for tell thresholds and durations
TELL_MAX_TTL = 30
SIGNIFICANT_BPM_CHANGE = 8
LIP_COMPRESSION_RATIO = 0.35
EYE_BLINK_HEIGHT = 0.15

def new_tell(text, ttl=TELL_MAX_TTL):
    """
    Creates a new tell object with a text description and time-to-live (TTL).
    """
    return {'text': text, 'ttl': ttl}

def decrement_tells(tells):
    """
    Decreases the TTL of all tells in the dictionary and removes those that have expired.
    """
    tells_copy = tells.copy()
    for key, tell in tells_copy.items():
        tell['ttl'] -= 1
        if tell['ttl'] <= 0:
            del tells[key]
    return tells

def detect_bpm_change(recent_bpms, avg_bpms):
    """
    Detects significant changes in BPM (Beats Per Minute) and returns appropriate tells.
    """
    bpm_display = "BPM: ..."
    bpm_change = ""
    recent_avg_bpm = np.mean(recent_bpms) if len(recent_bpms) > 1 else 0
    avg_all_bpm = np.mean(avg_bpms) if len(avg_bpms) > 1 else 0

    if abs(recent_avg_bpm - avg_all_bpm) > SIGNIFICANT_BPM_CHANGE:
        change = "increasing" if recent_avg_bpm > avg_all_bpm else "decreasing"
        bpm_change = f"Heart rate {change}"
        bpm_display = f"BPM: {int(recent_avg_bpm)}"

    return bpm_display, bpm_change

def assess_gaze(gaze_measure):
    """
    Evaluates the gaze measure to detect significant changes indicating deception.
    """
    # Placeholder for logic to assess significant gaze changes
    if gaze_measure > 0.5:  # example threshold
        return "Change in gaze direction"
    return None

def evaluate_blinking(blinks):
    """
    Evaluates blinking rate to identify increased or decreased blinking related to deception.
    """
    recent_blink_rate = np.mean(blinks[-10:])  # Last 10 frames
    overall_blink_rate = np.mean(blinks)

    if recent_blink_rate > 1.5 * overall_blink_rate:
        return "Increased blinking"
    elif overall_blink_rate > 1.5 * recent_blink_rate:
        return "Decreased blinking"
    return None

def check_lip_compression(lip_ratio):
    """
    Checks the lip compression ratio to detect signs of stress or deception.
    """
    if lip_ratio < LIP_COMPRESSION_RATIO:
        return "Lip compression observed"
    return None

def update_tells(tells, new_tells):
    """
    Updates the existing tells dictionary with new tells, resetting the TTL if already present.
    """
    for tell, text in new_tells.items():
        if text:
            tells[tell] = new_tell(text)
    return tells

def process_deception_indicators(face_data, hand_data, metrics):
    """
    Processes various metrics to determine possible deception indicators.
    """
    tells = {}
    # Process each metric to check for deception indicators
    if 'bpm' in metrics:
        bpm_display, bpm_change = detect_bpm_change(metrics['recent_bpms'], metrics['avg_bpms'])
        tells = update_tells(tells, {'bpm_display': bpm_display, 'bpm_change': bpm_change})
    if 'gaze' in metrics:
        gaze_tell = assess_gaze(metrics['gaze'])
        tells = update_tells(tells, {'gaze': gaze_tell})
    if 'blinks' in metrics:
        blink_tell = evaluate_blinking(metrics['blinks'])
        tells = update_tells(tells, {'blinking': blink_tell})
    if 'lip_ratio' in metrics:
        lip_tell = check_lip_compression(metrics['lip_ratio'])
        tells = update_tells(tells, {'lip_compression': lip_tell})

    return tells

def get_area(image, draw, topL, topR, bottomR, bottomL):
  topY = int((topR.y+topL.y)/2 * image.shape[0])
  botY = int((bottomR.y+bottomL.y)/2 * image.shape[0])
  leftX = int((topL.x+bottomL.x)/2 * image.shape[1])
  rightX = int((topR.x+bottomR.x)/2 * image.shape[1])

  if draw:
    image = cv2.circle(image, (leftX,topY), 2, (255,0,0), 2)
    image = cv2.circle(image, (leftX,botY), 2, (255,0,0), 2)
    image = cv2.circle(image, (rightX,topY), 2, (255,0,0), 2)
    image = cv2.circle(image, (rightX,botY), 2, (255,0,0), 2)

  return image[topY:botY, rightX:leftX]

























from metrics import *
import threading
from config import *

LIP_COMPRESSION_RATIO = .35 # from testing, ~universal
TELL_MAX_TTL = 30 # how long to display a finding, optionally set in args
calculating_mood_lock = threading.Lock()


def process(image, face_mesh, hands, calibrated=False, draw=False, bpm_chart=False, fps=None):
  global tells, blinks, hand_on_face, calculating_mood_lock
  tells = decrement_tells(tells)

  face_landmarks, hands_landmarks = find_face_and_hands(image, face_mesh, hands)
  if face_landmarks:
    face = face_landmarks.landmark
    face_area_size = get_face_relative_area(face)

    if calculating_mood_lock.acquire(blocking=False):
        try:
            emothread = threading.Thread(target=get_mood, args=(image,))
            emothread.start()
        finally:
            calculating_mood_lock.release()

    # TODO check cheek visibility?
    cheekL = get_area(image, draw, topL=face[449], topR=face[350], bottomR=face[429], bottomL=face[280])
    cheekR = get_area(image, draw, topL=face[121], topR=face[229], bottomR=face[50], bottomL=face[209])

    avg_bpms, bpm_change = get_bpm_tells(cheekL, cheekR, fps, bpm_chart)
    tells['avg_bpms'] = new_tell(avg_bpms) # always show "..." if BPM missing
    if len(bpm_change):
      tells['bpm_change'] = new_tell(bpm_change)

    # Blinking
    blinks = blinks[1:] + [is_blinking(face)]
    recent_blink_tell = get_blink_tell(blinks)
    if recent_blink_tell:
      tells['blinking'] = new_tell(recent_blink_tell)

    # Hands on face
    recent_hand_on_face = check_hand_on_face(hands_landmarks, face)
    hand_on_face = hand_on_face[1:] + [recent_hand_on_face]
    if recent_hand_on_face:
      tells['hand'] = new_tell("Hand covering face")

    # Gaze tracking
    avg_gaze = get_avg_gaze(face)
    if detect_gaze_change(avg_gaze):
      tells['gaze'] = new_tell("Change in gaze")

    # Lip compression
    if get_lip_ratio(face) < LIP_COMPRESSION_RATIO:
      tells['lips'] = new_tell("Lip compression")


  return 1 if (face_landmarks and not calibrated) else 0

          
def decrement_tells(tells):
  if not tells:
    return tells
  for key, tell in tells.copy().items():
    if 'ttl' in tell:
      tell['ttl'] -= 1
      if tell['ttl'] <= 0:
        del tells[key]
  return tells

def new_tell(result):
  return {
    'text': result,
    'ttl': TELL_MAX_TTL
  }