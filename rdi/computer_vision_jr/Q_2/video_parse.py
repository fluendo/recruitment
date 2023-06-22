import cv2

####### PARAMETERS ##########

# Define the video name
video_name = 'test_video.mp4'

# Define extracted frames save folder
frame_saving_folder = 'ultralytics/datasets/football_pre/images/'

# Define extracted frames root name
frame_root_name = 'im'

##############################

if __name__ == '__main__':
  # Import the video
  vidcap = cv2.VideoCapture(video_name)
  # Read first frame
  success, image = vidcap.read()
  # Define the intial frame number
  count = 0
  # Iterate the frames of the video
  while success:
    # Save frame
    cv2.imwrite(frame_saving_folder+frame_root_name+str(count)+'.jpg', image)
    # Read next frame
    success, image = vidcap.read()
    # Increase frame number
    count += 1
  print('Extracted %d frames' % count)
