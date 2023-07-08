# 1. What do you think it occurred during this model development (training & evaluation)? 

One of the issues could involve the training phase and the dataset used. The model was probably trained to identify and track the football using as reference the color, shape and its movement. On the articles we can see the model confuse the bald linesman's head with the characteristics that it learnt to recognize on the football, since these do share similarities (color, shape).

Another issue could be on the evaluation phase where the model's performance may not have been thoroughly tested in different scenarios, like having a bald linesman. 

And a last problem could be the quality of data on the training and evaluation datasets, these not having enough relevant data for the model to learn how to differenciate objects with similar characteristics.

# 2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks

Firstly, I would improve the training data by adding more examples of relevant images, like bald linesman and players, more types of balls... This way we can help the model improve when differenciating between the ball and individuals. A drawback would be the effort in collection or generating diverse training data.

Lastly, it would be great to incorporate multi-object tracking, to help the model track multiple objects at the same time and differenciate them. This would help with better tracking but it would increase the complexity of the model and may require more data.

# 3. Extra: Do you know any tracking algos (Deep learning based) that could be used here?

I do not know any tracking algos, but I think an algo which uses a CNN architecture would be a great fit for this case.