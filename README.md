# Hingebot
## Animation
<img src="hingebot_1.gif"/>

## Parametric Design Program
Click here to run "hingebot.ipynb" in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RichardPotthoff/hingebot/blob/main/hingebot.ipynb#scrollTo=Design_Form)

Click here to run "hingebot.ipynb" in Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RichardPotthoff/hingebot/main?labpath=hingebot.ipynb)
### Tripod Design
<img src="tripod_design.jpg"/>

<img src="tripod_preview.png"/>

### Capstan Design
<img src="capstan_design.jpeg"/>

<img src="capstan_preview.png"/>

# Photos
## Assembled Hingebot 
<img src="hingebot_assembled.jpeg"/>

## Detail: Continuous Cable Capstan
<img src="capstan_single_cable.jpeg"/>

## Detail: Split Cable Capstan
<img src="capstan_split_cable.jpeg"/>

# Configuration / Calibration
<img src="hingebot_calibration.jpeg"/>

## Mechanical preparation
* The cables must be parallel to the axes of the coordinate system with the toolhead at the origin of the coordinate system.
* The vertical slope of the cables must patch the pitch/circumference ratio of the capstan grooves. (1.25mm/60mm)
* The vertical position of the capstan must be adjusted to match the vertical position of the cable, so that the cable does not rub against the flanks of the groove.
 
## Parameters for printer.cfg
* rotation_distance: 60 ;capstan circumference in mm
* anchor: -350 ;position of capstan/cable tangent point, mm from the origin (+350 for y)
* kinematics: hingebot ;("winch" may be used if "hingebot" is not available)
