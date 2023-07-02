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

#### 1. What do you think occurred during this model development (training & evaluation)? 
Without any previous research apart from the links provided and the assumption that the model was not built using traditional computer vision methods rather than deep learning, I assume that the goal of the model was object recognition (detection) which entails 2 main tasks, image classification, and object localization. The model experienced confusion due to the visual similarities between the linesman and the ball, such as their shared color or shape, which might have labeled the bald linesman with higher accuracy than the ball for the correct class, especially when the view was obscured by players or shadows. This happened most likely due to not enough diversity in the dataset that resembled bald people.

#### 2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks
This error is what Chanda et. al call a "Logic in AI focused on the non-salient part of input information" where the AI system is provided all the information it needs and yet it makes a judgment based on the less-salient part of the information it receives [1]. In our case, the reason for misjudgment was the presence of a non-salient pattern resembling the pattern of interest. 

While my first thought roamed around the idea of applying data augmentation or adding more features that describe the objects better to the training data to fix the bias, my final conclusion relies more on the statement that the approach is not the most suitable for this task. Nonetheless, data augmentation could introduce more diverse examples that emphasize the distinction between the linesman and the ball. Some drawbacks of this are that collecting and labeling additional training data can be time-consuming and expensive and it might be challenging to find representative data that resembles accurately and generalizes well.

For a more novel approach, in order to fix this issue we could implement multi-task learning where, instead of using a live object detection approach, we could further incorporate object tracking with velocity control such as that mentioned in "Soccer Ball Tracking Using Dynamic Kalman Filter with Velocity Control" [2] where their algorithm includes needless objects elimination and takes into consideration the dynamic conditions of the ball. This would result in a direct advantage since the linesman would be omitted from the frames and the only relevant objects would be the players and the ball. Furthermore, appearance-based features such as color, size, shape, etc, become irrelevant and rely on Kalman Filter-based features, making the model more robust to distortion and occlusions. On the contrary, this result might be time-consuming and computationally expensive which would require computational power for live tracking. (Could also just include needless object elimination for a faster fix).  

#### 3. Extra: Do you know any tracking algos (Deep learning based) that could be used here?
From the top of my head, I can only recall YOLOv7 and SORT, which might be useful for this problem.  
![image](https://github.com/yibeisita/recruitment/assets/74725809/b4ba7ee0-3913-4c5b-8484-0fa0c2ee8fa2)


#### References
[1] Chanda, S.S., Banerjee, D.N. Omission and commission errors underlying AI failures. AI & Soc (2022). https://doi.org/10.1007/s00146-022-01585-x
[2] J. -Y. Kim and T. -Y. Kim. Soccer Ball Tracking Using Dynamic Kalman Filter with Velocity Control (2009). Sixth International Conference on Computer Graphics, Imaging and Visualization, Tianjin, China, 2009, pp. 367-374, doi: 10.1109/CGIV.2009.87.


