from ultralytics import YOLO
import cv2

####### PARAMETERS ##########

# Define the dataset name, which should be saved on the ultralytics/datasets folder
dataset_name = 'football'

# Define the dataset config file name, which is inside the dataset folder
dataset_config = 'data.yaml'

# Set the image/directory for prediction
predict_path = 'im92.jpg'

# Define if the model is trained with the specified dataset before prediction
train = False

# Weights used for prediction/training, used 'yolov8n' or any other for the training
pretrained_model = 'football.pt'

##############################


if __name__ == '__main__':

    # Compute prediction path
    predict_save_path = '/'.join(predict_path.split('/')[:-1]) + ('/' if len(predict_path.split('/'))>1 else '')

    if train:
        # Load a pretrained model
        model = YOLO(pretrained_model)
        # Train the model
        model.train(data=dataset_name+dataset_config, epochs=30, lr0=0.01, lrf=0.01)
        # Extract training metrics
        metrics = model.val()  # evaluate model performance on the validation set
    else:
        # Load the model with the trained weights
        model = YOLO(pretrained_model)

    # Make the prediction
    predictions = model(predict_path)

    # Visualize and safe predictions
    for pred in predictions:
        img_pred = pred.plot()
        img_name = pred.path.split('/')[-1]
        cv2.imwrite(predict_save_path+'pred_'+img_name, img_pred)