This file was written by a human as an initial sketch for the system.  Agents should not edit this file.

I like to stream video on my laptop.  A particular case is streaming sports while I cook or attend to other tasks.  In this case my hands are occupied and I can't mute the audio during an ad.  I want to build a lttle system that runs in the background and watches my screen.  When it detects that an ad is playing, it automatically mutes the audio, and similarly unmutes when the ad is done.  

Below is a sketch for implementing such a system.

* Phase 1: initial proof of concept

The first thing we need is a tool that will grab a substantial number of frames (say in the dozens to hundreds) that will be hand labeled by me.  I'll start this tool in the background and let it run while watching a stream.  It will grab a screenshot and store it in a folder named by the timestamp.  I've been thinking of the frequency of screenshots.  Most ads are 30 seconds, I think, so a prime number under 30 seems like a good default interval, perhaps 23 seconds.  During a 2 hour stream, we would get 469 screenshots if my math is correct.  

Next we need a tool that a human can use to hand label the data.  I'm imagining a grid of screenshots, each an inch or two wide, just big enough that one could easily recognize which class they are in.  The user would click on an image to add/remove it from a selection, then press a key (maybe a for ad, n for not ad) to classify them.  This would get stored in some presistent store, maybe a sqlite database, but whatever seems reasonable. 

Finally we need to train a classfier.  I'm imagining feeding the images into some pre-trained image classification net and taking the last-layer activations and using them to train a classfier (random forest or GBM would be my first choice, but whatever works).  We'd use n-fold cross validation to get classifier accuracy statistics.  My assumption is that a sporting event looks very same-y most of the time (green grass, score bug or whatnot) and that we'll get very high classification accuracy on that part of the stream showing the main action.  Commentary or other pre-recorded bits of the show might be harder, but we'll see.

The classifer will need to serialize itself so that we can do inference in phase 2.

An open question is what resolution to store the screenshots at.  Is disk usage a major concern?  I haven't done the math.  `df -h .` reports 141Gi available.

* Phase 2: the actual system

Now that we have a classifier, we build a system that samples the screen every few seconds.  Each screenshot is classified as game/not-game and when the score is below some threshold the audio is muted.  Now we might sample more frequently, so as not to remain muted too long when action resumes, but simimlarly, when we think it is likely that we're back to the game, we unmute (having recorded the prior volume).  If the user modified the volume on their own (the current volume does not match what we set it to when muting), then leave it alone.

This part of the system might choose to write difficult to classify frames out for later hand classification using the tool described in phase 1.

Update Feb 28, 2026:

It occurs to me that during the operation of the phase 2 monitor, I need a way to inspect it's operation and correct errors.  Here's what I'm imagining.  The monitor has a simple GUI.  When the monitor app is in the foreground it's operation is suspended, it only takes screenshots and mutes/unmutes when it's window doesn't have focus.

The monitor stores and displays the last few screenshots, maybe a minute or two's worth.  If I'm watching a stream and the classifier makes a mistake, I can switch over to the monitor, find the frame it misclassified and hand classify it right then and there.  Then swap back to the content and let it keep going.

We might leverage the UI of the labeler for this or write something completely new, whatever makes the most sense.