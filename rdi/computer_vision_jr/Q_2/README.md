# Question #2


## Computer Vision knowledge

### Problem description:

In Fluendo we are developing an AI-based system to automatize some sport analysis tools in order to help experts as coaches or scouts to identify a team's strong & weak points during a game or training session from a set of recorded images/videos.


Typical use-cases:

- Team heat-maps drawing: 

![heat](../imagery/heat.jpg)

- Virtual Assistant referee tools:

![var](../imagery/var.jpg)

- Player action recognition:

![expl](../imagery/expulsion.jpg)






In order to provide these smart functionalities one of the most important milestones is to achieve an accurate **team detection & classification system** which allow us to distinguish among players of different teams and discard referees and coaches. 

In this test you will be asked about this last functionality (see below image as reference)

![expl](../imagery/team.jpg)

## Test:

### Assumptions:

- Both traditional and modern (Deep Learning) techniques can be proposed

  

### Questions:

Please provide a document (and extra data if any) answering each question below.

1. Which algorithms would you use to achieve this? Please provide at least 2 proposals

The first approach could be using HoG descriptors to detect people in the image and then use the colour information inside the region, with some filters, to make the classification. This approach needs to adjust the filters according to the colours of the team/referee.

Another approach that relies less on human intervention could be approach is to use [YOLO](https://github.com/ultralytics/yolov5). This could be used to detect and classify people according to their cloths. However, particular attention should be paid to the cases where the teams/referees use very similar colors. 

2. Which potential bottlenecks may this problem encounter? (i.e. non-static camera). Just list them (if any)

In this case, weather/lighting conditions can affect the classification.

Another problem may be people overlapping, which can cause a mis-classification.

3. Could your proposals be scaled to other team-based sports? (i.e. hockey, rugby, ...) Which would be worst-case scenarios?

There should be no problem for the cases where the teams have some color distinction. For the case of the handcrafted method, maybe the descriptors should adapt more to the new common people poses. 

The worst case would be near identical cloths with very small differences, in which the system will struggle to make the classification.

3. Extra: provide a sample code with some proposal running
