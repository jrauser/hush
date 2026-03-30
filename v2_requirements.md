
# Introduction

Version 1 of the system has been working very well.  For v2, I'd like to unify the system into a smaller number of compenents and be much more trageted in how we capture screenshots.  I'd also like to put on an additional layer of UI polish.

V2 will have two major components:

1. A chrome extension that can capture video from a tab rather than screenshotting the whole screen.  This allows the user to use different apps at will and the system is only looking at the one tab that is streaming.  The extension can mute the audio for that one tab rather then using a system CLI tool.

2. A single unified app for capture, labeling, training and live monitoring.  The extension will send frames to the app and take configuration info from the app. 

# The app

The app has two major modes of operation: 1) bulk capture and labeling, and 2) live monitoring.  It has a third auxilliary operation: re-training the model.

## Bulk capture and labeling

In this mode, the capture interval should be a prime number of seconds and not too small (say no more often than 7 seconds?), so as not to fill up the disk needlessly.  We want more variety.  

Stored images can be selected by selected by class (unlabeled, ad, content, etc.) and labeled by selection and keystroke as in v1.

## Live monitoring

Same basic premise as in V1.  Capture interval is short, default 2 seconds.  There is a pause/resume button.  In active operation there is a hysteresis loop controlling muting/unmuting.  Frames stream by but aren't stored unless relabeled.  Same selection/labeling interface as in the bulk capture mode.  

## Model selection and training

Upon demand, spawn a thread to retrain the model.  I don't know where this control lives.  A button?  A "models" dropdown?  When complete, the user can view model diagnostics, ROC curve.

There should be a way to users browse available models and choose one to use.  I'm not sure how to handle this.  Maybe just a button that spawns a new window with available models in a column on the left, and when the user picks one, model diagnostics appear in a pane on the right.

## Tech stack

After some research, PyQt6 seems like a good library, but I'm open to suggestions.

Flask or FastAPI?  I think FastAPI.

The app offers an http endpoint the extension can send caputres to.

The app also offers a web socket interface that the extension can connect to for control plane messages.

We'll use the same SQLite database as V1.  I don't see a need to change that.  Frames can still be stored in the filesystem as V1 does.

## Control plane

The app sends messages via web socket to the extension to control its behavior.

* Enter active mode (send frames on an interval).  Parameters: capture interval X seconds
* Enter idle mode (stop sending frames)
* Mute/Unmute audio


# The extension

To enable operation, the user can click on the extension's icon which connects to the app.  If the extension can't connect it tells the user.  The extension is configured (capture frequency, operation mode, etc.) from the app.  The extension's icon should change color based on whether it is connectes to a running instance of the app.

I'd like it if the extension could tell the user about it's current configuration state: idble to capturing, what interval, muted or not.  If the user wants to override the mute could they do it from the extension or the app or both?

## The control plane

Upon enablement, the extenstion extablishes a web socket with the app.  See control plane section above for details.

## Screenshots

Screenshots are posted to an endpoint offered by the server.  Need a plan for  graceful degradation if the server gets bogged down, hangs or vanishes altogether.  


# Installation

I'd like the app to be installable for normal people.  Installation would be download and install the extension, and download and install the app.  I think these have to be separate?? Maybe the extension comes down with the app download and there are instructions for installation when you first boot the app?