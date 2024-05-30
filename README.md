# A Hardware-in-the-Loop Star Tracker Test Bed Thesis Code
This code is written in support of the star tracker test bed thesis project. For 
the thesis document see the Cal Poly Digital Commons.

# How to Run
System parameters that may be modified based on needs are presented in 
systemParameters.py. If parameters are modified, set regenerateDatabase to 
True to ensure the databases used by the other parts of the code contain the 
most recent information. 

To run the system in a pure software loop, bypassing the need for a camera and 
associated hardware, run the software.py file. numTrials sets the number of 
random attitudes that are generated and tested. 

To run the full system with hardware, run the hardware.py file with the 
display_img.png file open on the screen that is imaged by the camera. The focal 
length of the camera may need to be recalculated.  
