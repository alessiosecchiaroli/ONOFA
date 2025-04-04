import cv2 as cv
import numpy as np

def video_maker(ref_frame,work_frame):

    shape1 = ref_frame.shape
    shape2 = work_frame.shape

    if shape1 != shape2:
        raise ValueError("Shapes are different")

    if len (ref_frame.shape) == 2:
        ref_frame = cv.cvtColor (ref_frame, cv.COLOR_GRAY2BGR)
    if len (work_frame.shape) == 2:
        work_frame = cv.cvtColor (work_frame, cv.COLOR_GRAY2BGR)

    height, width, layers = ref_frame.shape
    fourcc = cv.VideoWriter_fourcc (*'mp4v')  # or 'XVID', 'MJPG', etc.
    video = cv.VideoWriter ('output.mp4', fourcc, 1, (width, height))  # 1 FPS
    # # in case, you want to see the video 0.5 FPS is better as the video is slower and more appreciable
    # video = cv.VideoWriter ('output.mp4', fourcc, 0.5, (width, height))  # 0.5 FPS


    # write the frames
    video.write (ref_frame)
    video.write (work_frame)

    video.release ()
    print ("Video saved as output.mp4")

    # # optional: play the video
    # cap = cv.VideoCapture ('output.mp4')
    # while cap.isOpened ():
    #     ret, frame = cap.read ()
    #     if not ret:
    #         break
    #     cv.imshow ("Video", frame)
    #     if cv.waitKey (1000) & 0xFF == ord ('q'):
    #         break
    # cap.release ()
    # cv.destroyAllWindows ()
