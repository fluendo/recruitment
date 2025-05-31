# Question #1


## Deep learning analytical thinking

### Problem description:

In 2024, several UK drivers were wrongly fined up to £105 for failing to pay the Dartford Crossing toll, despite never having used the crossing. The issue stemmed from ANPR cameras misreading vehicle licence plates, leading to incorrect penalty charge notices being issued. For instance, one driver received fines for vehicles with plates differing by a single character from her own, and another was fined for a vehicle of a similar make and color but with a slightly different registration number.

<div style="text-align: center;">
  <img src="../imagery/plates.png" alt="Plate 1" width="20%" style="display: inline-block; margin-right: 2%;">
  <img src="../imagery/ocr_plates.png" alt="Plate 2" width="20%" style="display: inline-block;">
</div>


#### Interesting Links:
- [Drivers wrongly fined £105 after number plate cameras get details wrong](https://www.inyourarea.co.uk/news/drivers-wrongly-fined-105-after-number-plate-cameras-get-details-wrong)
- [Drivers hit with Dartford Crossing fines despite never using it amid number plate camera issues](https://www.gbnews.com/lifestyle/cars/drivers-fined-dartford-crossing-never-used)


### Assumptions:
- System was being built with a Deep Learning based model (not traditional computer vision methods) 


### Questions:

Please provide a document (and extra data if any) answering each question below; please note that no practical exercises are required here (just theoretical analysis) but feel free to add any.

1. What do you think it occurred during this model development (trainning & evaluation)?

   Several issues might have happened during the development. The article mentions that : "The number plates were identical except for the last letter – hers is a G, whereas the last letter on this plate was a C."

   This is an interesting clue. If we notice that this keeps happening, if the model outputs a G instead of a C multiple times, it might be due to a dataset imbalance. The model might be trained on licence plates that contain more Gs than Cs and therefore is more comfortable predicting a G.

The second thing that might have happened is if the model was trained on clean and nice images of licence plates and that now it does more poorly on real world conditions where the images are more blurry.

   
2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks

   From reading the articles provided, it seems that the problem comes from blurry photos. My first reflex would be to use cameras with a faster shutter speed so that the photos taken will be less blurry even if the car is going fast. The drawback here would probably be cost, as you would need a large lens to make sure that the camera gets enough light during the short shutter period.

If the source of the problem is that the model was trained on clean images and not real world ones, a solution would be to use domain adaptation. The best way to do this would be to have annotated data from real world examples, but that might not be the case. If so, then the previous dataset can be used with some processing to simulate real world conditions, like maybe blurring the image slightly. The model would be more robust to real world blurry examples, but it would also mean we need to retrain the model.

   
3. What do you think it will occur when running this AI in a different country with different plates formats? How would you ensure system accuracy?

   I think it depends on the model arquitecture. If the model is more end to end desgined and uses attention mechanism that detects the characteristics of british licence plates, it will not work. However, if the model is more modular and comprises of multiple steps for example, car detection, then licence plate detection, then crop the image to only have the plate and then apply a general OCR algorithm, it could still work in different countries. I would ensure accuracy by separating the model arquitecture this way and maybe pass the final detection through a lightweight LLM to be able to filter out unwanted potential information that the algorthm thought was a licence plate number. 

   
4. Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?

  Yes, a few OCR algorithms based on deep learning would work here. For example, CRNN could be directly trained on images of license plates by feeding cropped images of plates and predicting sequences of characters as explained previously.

Another good option would be using attention-based OCR methods like ASTER, which could be fine-tuned specifically on British license plate datasets to improve accuracy on blurry or challenging cases.

   
6. Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)

   For my semester thesis, I worked on a project about tracking the movements of surgeons' hands during trauma surgery. The goal was to digitally record and analyze these hand movements in 3D so that surgeons could train more effectively and improve their skills.
  
  I was responsible for the entire project, meaning I had to design the whole system myself. This involved setting up two cameras to film the surgeons' hands, detecting hand positions using MediaPipe, and combining the videos to reconstruct a 3D model of the hand movements.
  
  One of the main challenges was dealing with errors when the software incorrectly detected hands in places where there were none. To handle this, I created a custom Kalman filter-based solution, which predicted hand movements and filtered out these errors.
  
  Another difficulty was synchronizing the two cameras since they didn't capture frames at precisely the same time. This sometimes affected the accuracy of the 3D reconstruction. I managed this issue by minimizing timing differences through software adjustments, although using synchronized cameras would have been the ideal solution.
