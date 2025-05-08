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

At first, it might seem that the recognition system has been trained on a limited set of license plates, leading to the over-recognition of certain plates. Moreover, the system may
 not have been exposed to more complicated cases, such as disty plates, shiny plates...

Another possibility is that the system wasn’t trained to recognize complete plates, but rather individual characters. This could result in outputs with invalid plate formats,
 generating combinations that don't make sense.

2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks

In a first case, this behavior could be corrected by increasing the number of data and difficult cases. We could add dirty plates, plates with reflections, and vairations of different
 characters that are similar like 0 and O, or 1 and I.

In a second case, which may already be in use, we could identify, in addition to the license plate, other features of the car, such as the color, brand and model. So then the
information related to the license plate number could be referenced and it could be determined if the car is the correct one, and if the license plate is similar, the one imposed by
the characteristics could be returned.

3. What do you think it will occur when running this AI in a different country with different plates formats? How would you ensure system accuracy?

In different countries, license plates vary in length, order, type of characters, colors... The system, lacking different examples beyond those from the United Kingdom, would have a
 high probability of failing on foreign license plates.

Especially because of the fact that the characters are ordered differently and not all are in the same alphabet, since there are plates that contain Arabic characters...

4. Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?

The truth is that don't know much about OCR algorithms based on deep learning, but after researching, I've discovered an algorithm that could work. Tesseract, this is the character
 recognition algorithm that Google currently uses to recognize text in images, both typed and handwritten.

This algorithm is based on Long-Term Memory neural networks, which could be very useful for having a large database of different license plates. Not only having those that have been
 added during training, but also those of the cars that pass through the control.

5. Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)

As I have mentioned before, it's true that I am not an expert on this topic and therefore I have not worked on it previously. But this small test has given me the opportunity to
 search for information and learn about the topic, which I have found very interesting. Not only for the purpose, but also for how this entire system works from the inside.

