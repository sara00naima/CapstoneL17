# Scene defaults

STEM_TYPES = ["vocals", "drums", "bass", "guitar", "piano", "other"]

#simple positionings
FRONT = (0.0, 0.0)
LEFT  = (90.0, 0.0)
RIGHT = (-90.0, 0.0)
BACK  = (180.0, 0.0)

# Default spatial positions for each stem type in (azimuth_deg, elevation_deg).
# HARDCODED FOR NOW, PROBABLY SHOULD BE CHANGED LATER FOR USER DEFINED POSITIONING 
DEFAULT_POSITIONS_DEG = {
    "vocals": (0.0,    0.0),
    "drums":  (180.0,  0.0),
    "bass":   (0.0,  -20.0),
    "guitar": (-45.0,  0.0),
    "piano":  (45.0,   0.0),
    "other":  (0.0,   60.0),
}
