from ultralytics import YOLO

images_path = 'ultralytics/datasets/football_pre/images/'
labels_path = 'ultralytics/datasets/football_pre/labels/'
people_label = '0'
pretrained_model = 'yolov8n.pt'

model = YOLO(pretrained_model)  # load a COCO pretrained model
predictions = model(images_path)

for sample in predictions:
    boxes = sample.boxes
    img_name = sample.path.split('/')[-1].split('.')[0]
    f = open(labels_path+img_name+'.txt', 'w')
    label_data = ''
    for box in boxes:
        b = box.xywhn[0]  # get box coordinates in (x, y, width, height) format, normalized (0-1)
        c = box.cls
        c_str = str(int(c.tolist()[0]))
        if c_str == '0':
            b_str = str(' '.join(map(str, b.tolist())))
            label_data += c_str+' '+b_str+'\n'
    f.write(label_data)
    f.close()
