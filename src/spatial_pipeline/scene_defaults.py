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
    "vocals": (0.0,    20.0),   # front-centre, mid-elevation
    "drums":  (180.0,  20.0),   # back, raised into speaker dome (was 0° — below all speakers)
    "bass":   (0.0,    10.0),   # front-low, just inside speaker dome (was -20° — below all speakers)
    "guitar": (-45.0,  20.0),   # right-front
    "piano":  (45.0,   20.0),   # left-front
    "other":  (0.0,    53.0),   # overhead, matching the upper speaker ring
}
