
[Worker Safety Detection System]

!!!!!!    Status: Model is still not trained sufficiently (does not detect sufficiently 'person' class) !!!!!!

Used API and libs:

tensorflow==1.14.0
Object Detection API 

Used DL model:
Inception V2 Region-Based Convolutional Networks

Dataset:
VOC-2028 (https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)



Folders purpose:

1. Cam_Videos (contains videos from cams for a demo purpose)

2. Frames_Buffer (folder which contains frames with a one-second range)

3. inference_graph (directory for a graph location / model checkpoint)

4. training (model configuration and checkpoint path, it is required to extract a tf graph).
Link for my checkpoint to obtain pretrained weights: https://drive.google.com/open?id=11_8P8vIOYbEIIJOlvPz4xpHnY64QkUyN

5. Detection_cases (contains frames where inconsistencies are detected 
with a detection report)

This project is about making system that could detect 
safety norms violation. In our case, we are trying to detect
people with safety helmets (class 'hat') and people without 
them (class 'person').

Description

Imagine that you have a facility that requires some safety 
requirements that are critical to your staff. 
Usually companies require their workers to have it before
entering to the industry zone, but unfortunate accidents are 
inevitable due to some unexpected reasons. This problem becomes
especially significant on high-dangerous objects like 
construction sites. One way to deal with that is to place cameras 
throughout the facility. Still the problem is that it requires 
dozens (or hundreds) of cameras that stream videos is a nonstop 
manner. Obviously analyzing those videos are cumbersome and could 
require a lot of security staff. Even if a company could allow 
such enormous costs, it is still vulnerable to a human mistake. 
   One way to deal with that is to design a surveillance system 
that could be able to detect cases of non-compliance to safety norms.
So technically what happens is that you have a set of cams that 
send videos to some directory and save it according to a camera id.
Then you need analyze somehow these videos, but the problem is that
to make analysis effective without serious delay. Still it is not
that important to analyze the whole video, rather some set of frames
from it. For example, wee can take one-second frame from each video
and then analyze and make a report in a case of violation and send it 
to the security post together with a detection frame.




