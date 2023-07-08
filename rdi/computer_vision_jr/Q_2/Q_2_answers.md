# 1. Which algorithms would you use to achieve this? Please provide at least 2 proposals

After doing some research on some algorithms, CNN-based algorithms would be great. The first one would be DeepSORT [1, https://github.com/nwojke/deep_sort], great for tracking while assigning IDs to each object. The second one would be MDNet [2, https://github.com/hyseob/MDNet], which is a visual trackers based on a CNN.

# 2. Which potential bottlenecks may this problem encounter? (i.e. non-static camera). Just list them (if any)

Non-static camera, occlusion, lighting, noise in image, non-relevant data.

# 3. Could your proposals be scaled to other team-based sports? (i.e. hockey, rugby, ...) Which would be worst-case scenarios?

Yes, my proposals could be scaled to other team-based sports. The worst-case scenarios would involve the variations in player appearances, fast-paced actions, complex environments, and as mentioned before, occlusions.

# 4. Extra: provide a sample code with some proposal running

