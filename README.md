# Hingebot
## Animation
![Hingebot animation](https://github.com/RichardPotthoff/hingebot/blob/main/hingebot_1.gif?raw=true)
## Parametric Design Program
Click here to run "hingebot.ipynb" in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RichardPotthoff/hingebot/blob/main/hingebot.ipynb#scrollTo=Design_Form)

Click here to run "hingebot.ipynb" in Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RichardPotthoff/hingebot/main?labpath=hingebot.ipynb)
### Tripod Design
![Tripod Design](https://github.com/RichardPotthoff/hingebot/blob/main/tripod_design.jpg?raw=true)

![Tripod Design](https://github.com/RichardPotthoff/hingebot/blob/main/tripod_preview.png?raw=true)
### Capstan Design
![Capstan Design](https://github.com/RichardPotthoff/hingebot/blob/main/capstan_design.jpeg?raw=true)

![Capstan Design](https://github.com/RichardPotthoff/hingebot/blob/main/capstan_preview.png?raw=true)
# Photos
## Assembled Hingebot 
![Assembled Hingebot](https://github.com/RichardPotthoff/hingebot/blob/main/hingebot_assembled.jpeg?raw=true)
## Detail: Continuous Cable Capstan
![Continuous Cable Capstan](https://github.com/RichardPotthoff/hingebot/blob/main/capstan_single_cable.jpeg?raw=true)
## Detail: Split Cable Capstan
![Split Cable Capstan](https://github.com/RichardPotthoff/hingebot/blob/main/capstan_split_cable.jpeg?raw=true)

# Configuration / Calibration

![Split Cable Capstan](https://github.com/RichardPotthoff/hingebot/blob/main/hingebot_calibration.jpeg?raw=true)

## Mechanical preparation
* The cables must be parallel to the axes of the coordinate system with the toolhead at the origin of the coordinate system.
* The vertical slope of the cables must patch the pitch/circumference ratio of the capstan grooves. (1.25mm/60mm)
* The vertical position of the capstan must be adjusted to match the vertical position of the cable, so that the cable does not rub against the flanks of the groove.
 
## Parameters for printer.cfg
* rotation_distance: 60 ;capstan circumference in mm
* anchor: -350 ;position of capstan/cable tangent point, mm from the origin (+350 for y)
* kinematics: hingebot ;("winch" may be used if "hingebot" is not available)
