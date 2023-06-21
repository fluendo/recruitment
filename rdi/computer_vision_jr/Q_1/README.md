# Question #1


## Deep learning analytical thinking

### Problem description:

In 2020 when AI was doing its first steps on professional sports, the following incident became world-wide viral after one AI-based tracking system intended to follow a soccer ball got confused and started following a bald linesman instead.



![ref](../imagery/ref.jpg)



### Interesting links:

- https://www.theverge.com/tldr/2020/11/3/21547392/ai-camera-operator-football-bald-head-soccer-mistakes
- https://indianexpress.com/article/trending/trending-globally/ai-camera-mistakes-referees-bald-head-for-ball-follows-it-through-the-scotland-match-6911260/
- https://www.youtube.com/watch?v=SPbTKfu0zUY





## Test:

### Assumptions:

- System was being built with a Deep Learning based model (not traditional computer vision methods) 



### Questions:

Please provide a document (and extra data if any) answering each question below; please note that no practical exercises are required here (just theoretical analysis) but feel free to add any.

1. What do you think it occurred during this model development (trainning & evaluation)? 

The dataset had few/none bald people, thus the system was unable to learn that a person could have a ball-like shape in the head.
Additionally, in some other areas, there have also been cases where the detection of small objects has been mismatches with other small objects, specially when shapes and colors match.

Another possibility, is that the system in charge of the tracking was not prepared to handle multiple ball in the fild and when the head was detected as a ball the focus switched.

2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawback

One method is to constrain the bounding box displacement between frames, as it is known that humans have strength limitations, that can generate an upper bound to the speed ob the object being tracked.
Seeing the example the camera changed position in such a way that was physically impossible for the ball to reach in this certain time. However, despite being a simple approach that can be implemented easily, it can fail when the bald person's head is near the ball, as this tracking transition would be feasible in the specific time.

The other approach implies adding more examples of bald people playing in the dataset, this can help the system learn better when the object is a ball or not. Additionally, a new bald head class can be included, to specifically force the system to learn what is a bland head and a ball.
This approach can be more time-consuming and difficult to implement, as new labeled data needs to be added to the training set and retraining the model, and bald people playing data could be hard to find. 

Finally, both proposes could be combined to obtain an even better result.

3. Extra: Do you know any tracking algos (Deep learning based) that could be used here?

The most typical approach is to use [YOLO](https://github.com/ultralytics/yolov5) to detect the object you want to track (or other object detection architecture). However, in the case where the same object appears multiple times, this can lead to tracking errors (for the case of football, as only one ball is at the filed, this could work great).
When looking at the new version description, [YOLOv8](https://github.com/ultralytics/ultralytics) , I found that is also able to do tracking of individual objects, with good results.

There are other architectures that handle multiple objects at once, such as [TrackFromer](https://github.com/timmeinhardt/trackformer). However, transformer models tend to need more data and computational resources to train. Although I know transformer tracking methods perform good, I have not yet had chance to work with them 😢.