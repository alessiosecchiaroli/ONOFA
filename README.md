ONOFA - ORCHID Nozzle Optical Flow Algorithm

This is a code for calculating the background deflection due to density gradient in the ORCHID nozzle, by means of Background Oriented Schlieren (BOS) and Optical Flow algorithm.

The code in this folder is mainly working with OpenCV library (cv2).

It is tuned for the ORCHID nozzle, so a few parameter might be hardcoded.

The input are two frames (working  picture and reference picture), it finds the position of the crosses on either sides of the nozzle.
It overlap the two frames, to correct for the rigid shift of the nozzle between the two frames.
After the nozzle shape in the frame is overlapped, a mask is generated, and finally the optical flow algortihm is used to obtain the displacement.






