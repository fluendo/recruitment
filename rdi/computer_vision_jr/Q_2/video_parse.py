import cv2

video_name = 'test_video.mp4'
frame_saving_folder = 'ultralytics/datasets/football_pre/images/'
frame_root_name = 'im'

vidcap = cv2.VideoCapture(video_name)
success, image = vidcap.read()
count = 0
while success:
  cv2.imwrite(frame_saving_folder+frame_root_name+'%d.jpg' % count, image)
  success, image = vidcap.read()
  count += 1
print('Extracted %d frames' % count)
