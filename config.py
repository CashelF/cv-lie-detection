class Config:
    TELL_MAX_TTL = 30 # how long to display a tell
    MAX_FRAMES = 120 # modify this to affect calibration period and amount of "lookback"
    FPS = 30 # default fps
    RECENT_FRAMES = int(MAX_FRAMES / 10) # modify to affect sensitivity to recent changes