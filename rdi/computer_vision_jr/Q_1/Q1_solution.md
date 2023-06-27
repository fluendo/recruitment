# Question 1

## 1. What do you think occurred during this model development (training & evaluation)?

In the incident you described, where the system intended to follow a soccer ball but got confused and started following a bald linesman instead, there are several possible factors that could have contributed to this error during the model development, training, and evaluation process.

In this case, I think that the issues were due to the quality of the training set. Also it didn’t help that the camera angle made it appear as though the linesman was inside the boundaries of the lines on the field and the similarities in the ball’s color and the referee’s head.

The training data used to train the model might have been biased and lacked sufficient examples. As a result, the model might not have learned to distinguish between the ball and the linesman accurately. The model might not have been exposed to a diverse range of scenarios during training. If the training dataset primarily consisted of ideal situations with clearly visible balls, where the linesman and the ball were well-separated and easily distinguishable, it could have struggled when faced with ambiguous situations involving similar-looking objects, more challenging perspectives, occlusions, or instances where the linesman and the ball were in close proximity, or inside the boundary of the lines on the green field and had the same background. Even if the training data included examples of bald linesmen, it might not have been representative enough to encompass the wide range of scenarios that can take place during actual matches.

Additionally, the evaluation process plays an important role in assessing the model's performance and identifying potential shortcomings. The evaluation primarily focused on tracking the ball and did not thoroughly consider edge cases, so the issue of the model mistakenly following the linesman might have been overlooked. If the evaluation metrics only measured the model's accuracy in tracking the ball and did not include scenarios specifically designed to test its ability to ignore similar-looking instances, the problem of tracking the linesman, and the fact that the model was not able to generalize, would not have been discovered until it occurred during a live match.


## 2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks

To fix the behavior there are several potential options.

The first option can focus on enhancing the training data and retraining the model. Making a more diverse dataset, including more difficult scenarios, with different camera viewpoints and perspectives, lighting conditions and including occlusions, can help the model to learn to differentiate between  similar objects more effectively and avoid overfitting. 

Also, augmenting the training data with additional examples can help the model. Revisiting the training process with the augmented dataset allows the model to learn from the new samples and adjust its behavior accordingly.
Collecting and annotating a diverse dataset can be very time-consuming and require additional computational resources. If the augmented dataset is not carefully balanced and representative of real-world scenarios, there is a risk of overfitting the model, leading to poor generalization.

The second option is instead of training the model solely to track the ball, we could also incorporate a multi-object tracking approach that simultaneously considers multiple objects, such as the ball, players, and linesmen. This enables the model to better discriminate between different objects of interest. By trying to distinct features of the ball (shape, color)  and considering contextual cues, such as ball trajectory, the model can reduce the chances of mistakenly focusing on the linesman.
Implementing multi-object tracking and attention mechanisms adds complexity to the model architecture, which may require additional computational resources and expertise.
The success of this approach relies on having a diverse and well-annotated training dataset that encompasses various scenarios involving the ball, players, and linesmen. Acquiring such a dataset can be challenging and time-consuming.


## 3. Extra: Do you know any tracking algos (Deep learning based) that could be used here?

Deep learning-based tracking algorithms could be potentially used to address the tracking issue in this scenario.

Deep SORT (Deep Simple Online and Realtime Tracking) is a deep learning-based tracking algorithm that combines a deep feature extractor, utilizing convolutional neural networks, with the SORT algorithm. It incorporates a deep feature embedding network to encode appearance information and a Kalman filter-based tracking framework to handle the temporal dynamics of the tracked objects. Deep SORT has shown promising results in multi-object tracking scenarios and can be adapted to track the ball. [[1](https://www.intechopen.com/chapters/75342)]

Additionally, CenterTrack is also a tracking algorithm that introduces an object detection network followed by an online tracking algorithm. The key idea behind CenterTrack is to estimate an object's center point and associate it with the corresponding object across frames. By using the center points, CenterTrack aims to address the challenges of occlusion, scale variation, and object interactions that often occur in crowded scenes.

Another state-of-the-art tracking algorithm that can be proposed is TrackNet, a deep learning network for tracking high-speed and small objects in sports applications, that could be a suitable choice for accurate ball tracking, as it is trained to recognize the ball from a single frame but also learn patterns from consecutive frames. [[2](https://arxiv.org/abs/1907.03698)]


