from ultralytics import YOLO
import cv2

dataset_name = 'football'
dataset_config = 'data.yaml'
predict_path = 'im92.jpg' # 'ultralytics/datasets/football/test/images/'
train = False # If True, load a COCO pretrained model i.e. yolov8n
pretrained_model = 'football.pt'


predict_save_path = '/'.join(predict_path.split('/')[:-1]) + ('/' if len(predict_path.split('/'))>1 else '')

if train:
    model = YOLO(pretrained_model)  # load a pretrained model
    model.train(data=dataset_name+dataset_config, epochs=30, lr0=0.01, lrf=0.01)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
else:
    model = YOLO(pretrained_model)  # load a pretrained model

predictions = model(predict_path)  # evaluate prediction
for pred in predictions:
    img_pred = pred.plot()
    img_name = pred.path.split('/')[-1]
    cv2.imwrite(predict_save_path+'pred_'+img_name, img_pred)