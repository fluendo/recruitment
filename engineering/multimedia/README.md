# Multimedia Engineer Test

## What is this test about?
If you are reading this document, then you are in the process of becoming a new member of our team. Like any other company, at Fluendo we are looking for the best talent to join our team, but it’s always hard to do so just by reading a CV or in a 1-hour Q&A interview.

![hacker](https://raw.githubusercontent.com/fluendo/recruitment/master/content/hacker2.gif)

The goal of this test is to quickly understand how you would fit in our team by hacking on a very simple video player application using the same development stack and tools we use on a daily basis. The video player app uses the GStreamer framework for video playback and either Qt or Gtk for the UI, using the one you know the most. As you will see, the test itself is quite simple and depending on your skills you should finish it in around 2-4 hours. It will try to cover some important aspects to us:
 * Use of git
 * Working with an existing code base
 * Parallelism and threads
 * Clean code
 * Documentation
 * Understanding of User Stories
 * Whatever other thing you want to show us!

## The rules

We will provide you with a description of a new feature for the video player app, as it would be done by our Product Manager, and a list of User Stories describing its functionality.
You will create the new application and publish your work in a public repository using a service like github or bitbucket.
As you will see, the User Stories do not have any UI design, so design it as you want since it won’t be taken in account.
What we will value instead, is the ability to work with new API’s, your autonomy to interpret and implement User Stories, the use of good design patterns and your ability to split your work into commits and branches.

![hacker](https://raw.githubusercontent.com/fluendo/recruitment/master/content/hacker3.gif)

## Video Player new features

At Fluendo we have decided to improve this video player with a set of new features, which are provided as new user storiess from our PM's

### Report the duration of the clip to the user
As a user, I would like to know the duration of the video clip loaded.
The duration should be printed in the controls bar with the following format:

`Duration: HH:mm:ss.SSS`

The story is successful when the user loads [Tears Of Steel]( http://ftp.halifax.rwth-aachen.de/blender/demo/movies/ToS/tears_of_steel_720p.mov) and the player shows:
`Duration 00:12:14.167`

### Know the current position of the stream
As a user, I would like to know the current position of the stream in the video player.
The current position should be updated with an interval of 20 ms.
The current position will be displayed in the controls bar before the total duration with the same format. The expected output is:

`Position: HH:mm:ss.SSS  Duration: HH:mm:ss.SSS`

The story is sucessful when the user starts playing a clip and the current position is updated.

### Seek to a position in the stream.
As a user, I would to seek to a position in the stream in the video player using a slider.
Considering the slider has a range between 0 and 1, the value of the slider is the relative position of the stream, calculated like `current_position/duration`.
The position of the slider will be updated with the same interval as the position label.
When the user drags the slider it will change the position of the stream with a seek.
The final design including the slider should like like:

`Position: HH:mm:ss.SSS [-----------------|---------------] Duration: HH:mm:ss.SSS`

The story is successful when the user starts playing a clip and the slider is updated with the relative position.
The story is successful when the drags the slider and a seek is performed updating the position of the stream.

### Create a timeline with a snapshots preview
As a user I would like to visually preview the contents of the stream once its loaded, in the form of 10 thumbnails displayed in a timeline. Thumbnails are spaciated in time at 1/10th of the duration of the clip. The thumbnails generation should happen in background and the timeline should be updated asynchonously with the new thumbnails as they are generated.
Here is an example of the desired feature.

<img src="https://raw.githubusercontent.com/fluendo/recruitment/master/content/timelinepreview1.png" alt="enable-repl" width="300">
<img src="https://raw.githubusercontent.com/fluendo/recruitment/master/content/timelinepreview2.png" alt="enable-repl" width="300">

In the snapshot project you will find a sample code to extract thumbnails from a video file.

The story is successful when the user loads a new video file and the thumbnails timeline is progressively generated with 10 thumbnails.

It's time to start hacking!

![hacker](https://raw.githubusercontent.com/fluendo/recruitment/master/content/THEGif.gif)

## Video Player project

The Video Player project has 3 components:
  * videoplayer-qt: sample video player using the QT framework
  * videoplayer-gtk3: sample video player using the Gtk+3 framework
  * snapshot: sample application to get snapshots from a video file
  
The sources of the video player are taken from the GStreamer project examples and tutorials with the intention to provide a very basic starting point to start implementing new features for the test.

### Setup
On a machine with a Debian-like OS, install the following packages:

```
sudo apt-get install cmake libqt4-dev qtgstreamer-plugins libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libboost-dev libqtgstreamer-dev libgles2-mesa-dev gstreamer1.0-libav libgdk-pixbuf2.0-dev libgtk-3-dev
```

### Configure
The project uses a CMake build system. To configure it enter either videoplayer-qt or videoplayer-gtk3 depending on the toolkit of your choice and configure it with:
```
cmake CMakeLists.txt
```

### Build
Once configured type:
```
make
```

### Run
To run the app launch:
```
./videoplayer
```

The player starts without any video file open. Click the open button and select a video file.
