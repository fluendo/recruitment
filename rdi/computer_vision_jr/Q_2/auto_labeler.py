from ultralytics import YOLO

####### PARAMETERS ##########

# Define images path
images_path = 'ultralytics/datasets/football_pre/images/'

# Define labels saving path
labels_path = 'ultralytics/datasets/football_pre/labels/'

# Set the pretrained model path and name
pretrained_model = 'yolov8n.pt'

# People labels ID in the pretrained model
people_label = 0

##############################

if __name__ == '__main__':

    # Load the pretrained model
    model = YOLO(pretrained_model)

    # Generate the predictions
    predictions = model(images_path)

    #  Process the predictions within each image to generate the labels
    for sample in predictions:
        # Extract the bounding boxes information
        boxes = sample.boxes
        # Extract the image name
        img_name = sample.path.split('/')[-1].split('.')[0]
        # Crate the labels file fo the image
        f = open(labels_path+img_name+'.txt', 'w')
        # Initialize the labels data
        label_data = ''
        # Iterate the bounding boxes inside the image
        for box in boxes:
            # Extrac the bounding box info with YOLOv8 format [(x, y, width, height) normalized (0-1)]
            b = box.xywhn[0]
            # Extract the bonding box class
            c = box.cls
            # Convert the class to string
            c_str = str(int(c.tolist()[0]))
            # Check if the class belong to the desired class
            if c_str == str(people_label):
                # Convert the bounding box info to string
                b_str = str(' '.join(map(str, b.tolist())))
                # Add the bounding box info to the labels data
                label_data += c_str+' '+b_str+'\n'
        # Writhe the labels data to the labels file and close it
        f.write(label_data)
        f.close()
