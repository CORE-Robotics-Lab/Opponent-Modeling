# import cv2
# import numpy as np
# from utils import save_video

# path = 'image_2.png'

# frame = cv2.imread(path)
# new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# cv2.imwrite('figures/image_2_rgb.png', new_frame)

import cv2
import numpy as np
from utils import save_video


# path = 'short-success'
# path = 'dodge'
path = 'chase-scene'
# path = 'faux-corner-dodge'
# path = 'LR-success'
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(path + ".gif")

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

all_frames = []
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    all_frames.append(new_frame)
    # Display the resulting frame
    # cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

save_video(all_frames, "figures/" + path + "_new.mp4", fps=30)