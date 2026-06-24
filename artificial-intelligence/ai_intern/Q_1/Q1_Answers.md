# Answers
## Q1

With no exact knowledge on the data or the details on the training or evaluation, pinpointing the problem exactly is a bit tricky.

- First though is we've got an OCR level problem since we've got letter confusion, it seems that we didn't feed it or train it on hard situations meaning our data is not properly augmented by blurring a little the letters or making harder to detect.

- The other thing that comes to mind looking at the problem is "another was fined for a vehicle of a similar make and color but with a slightly different registration number". Maybe the segmentation is leaving some color pixels that the OCR is considering, that is of course if the OCR is operating on colored crops rather than grayscale.

## Q2

- At first we need to train the OCR model on more complex data and hard situation, making the model robust and ready to tackle any kind of problem or letter recognition, the downside for this is training requires more data, even if augmented, just to avoid over-fitting and acquiring data needs time and resources.

- Second we need to look at our segmentation work, and make it grayscale, so it becomes resistant to colors, but this really depends on the contrast, black letters and white background? We're in good hands, white letters on red background? The contrast won't be that good. 

- Last thing, it may not be related to the exact model but to the whole pipeline itself, we really need focus on reducing the false positives, by setting a strict threshold, missing a faulty car is far permissible than fining an innocent one.

## Q3

The model will run, but not as accurately, we're talking about different dimensions, colors, letter fonts and sometimes even language! I'm thinking here of a base model that has variety, trained on big data from different countries and different situations, let that be our starting point then for each client we fine-tune or use feature extraction. It's illogical that the same model can perform with the same high accuracy in Japan and US at the same time!.

## Q4

I know the core algorithm which is CRNN, I also know EasyOCR and LPRNet. What I see convenient here is LPRNet, it's an up and ready lightweight algorithm made specifically for number plate recognition, we know our target here so no need to use a general purpose ocr algorithm like easyocr or even go to the length of making our own CRNN.

## Q5

During my latest mission in my work, we had a sports streaming company that wants its youth and academy football matches that use just a single wide camera to be similar to the professional broadcasting matches, they wanted player recognition, player and ball tracking and virtual automatic pan, tilt and zooming.

The work is confidential so I need to be discreet with some details

My role was working on the recognition and tracking part, the first difficulty I found is the models licensing, since naturally one must find a good base model to fine-tune, that's just when I found that the ultralytics yolo family fall under the GPL license and most well annotated, broad and big sports datasets are non-commercial only, I finally went to the yolox family since they are under the apache license.

Then the camera quality was mediocre, the ball cannot be seen from far distances, I used tiling to further crop the image into pieces to make detecting the ball easier and used kalman filter to predict its trajectory to make the ptz work easier.

The final goal was to put the whole pipeline into a SBC, so optimization is critical, and 30 fps is non negotiable, the bottleneck was detection, so detecting every N frames and leaving the work to tracking was the optimal idea, decreasing the image dimensions was not an option since the ball in an 1080p is small enough.