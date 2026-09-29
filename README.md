# Drone Flight Training Simulator - City Edition

Python + Ursina (Panda3D). Everything is generated in code: no asset downloads.

## What's in the world
- Procedural city: road grid with lane markings, zebra crossings and stop lines, sidewalks,
  ~100 buildings (glass towers, houses), trees, street lights, distant hills, drifting clouds
- **Traffic**: 64 cars and buses that follow lanes, keep their distance and obey working traffic lights
- **Pedestrians**: 56 people with walking animation who wait for the crossing signal
- Real-time **sun shadows**, plus **DAY / SUNSET / NIGHT** (lit windows, street lights, headlights)
- Wind with gusts, battery (recharges on a pad), altitude-hold assist, minimap, rotor sound
- 8 rings (+100 each) placed through the street canyons
- Two landing pads: **HOME** (blue, in the park) and **PAD B** (magenta, on the tower roof)

## Controls
| Key | Action |
|-----|--------|
| SPACE / LEFT SHIFT | Climb / descend (throttle) |
| W / S | Pitch forward / back |
| A / D | Roll left / right |
| Q / E | Yaw left / right |
| F | ALT-HOLD assist on/off (manual throttle when off) |
| V | Wind on/off |
| N | Day / sunset / night |
| C | Camera: chase / FPV / orbit |
| R | Reset drone |
| T | New run (score + rings) |
| H | Help |
| ESC | Quit |

Assist mode (default) holds altitude and position for you. Turn it off with **F** for the
original manual-throttle challenge (hover is about 50% throttle).

## Graphics settings (top of `main.py`)
| Setting | Meaning |
|---------|---------|
| `SHADOWS` | real-time shadows on/off (biggest FPS cost) |
| `SHADOW_RES` | 4096 = sharp, 2048 = faster |
| `CARS_PER_LANE`, `NUM_PEDS` | traffic / crowd density |
| `USE_LIT` | False = safe mode without lighting shaders |

## Run
```
pip install -r requirements.txt
python main.py
```

## Build the .exe (Windows)
Double-click `build.bat` -> `dist\DroneFlightSimulator.exe`

## Rules of flight
- Touching down faster than 4.5 m/s or tilted more than 15 deg = crash; so is hitting a wall
- Rooftops are landable
- Pad landing score is 50 to 200 depending on distance from the pad centre

## Ideas for the team
Gamepad support, timed race mode, delivery missions (pick up on HOME, drop on PAD B),
weather (rain/fog), replay/ghost, more vehicle types, sound for cars.
