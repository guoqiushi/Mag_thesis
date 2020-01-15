# 3-D localization based on Magnetic field

This is my Master thesis, supervised by Sizhen Bian, at the group Embedded Intelligence of DFKI.

## Sensors & Device 

* IMU
* Transmitter
* Ultra Sound 


## Data
We collected data both indoor and outdoor. The indoor data is at the second floor in DFKI Kaiserslautern, and collected outdoor data at the 

## Approach

![](./figure/1.jpeg)

## Code

| file | description |
| ---- | -------- |
| phy_model.py | a physical model, predicting the position based distances to 3 transmitters |
| ml_model.py| using machine learning methods to predict the position|
| dead_reckoning.py |Integrating the IMU data by using Dead reckoning algorithem|   
| Kalman_filter.py| |
| performance,py | |

## Experiment Result
 
## Reference

* Integrated WiFi/PDR/Smartphone Using an Unscented Kalman Filter Algorithm for 3D Indoor Localization
* A Robust Indoor Positioning System based on Encoded Magnetic Field and Low-cost IMU
