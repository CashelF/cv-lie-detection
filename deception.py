from metrics import *
import threading
from globals import *

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