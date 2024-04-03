MAX_FRAMES = 120 # modify this to affect calibration period and amount of "lookback"

tells = {}
blinks = [False] * MAX_FRAMES
blinks2 = [False] * MAX_FRAMES # for mirroring

hand_on_face = [False] * MAX_FRAMES
hand_on_face2 = [False] * MAX_FRAMES # for mirroring

mood = ''