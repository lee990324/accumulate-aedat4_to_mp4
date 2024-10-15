import datetime

import dv_processing as dv
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Event accumulation sample.')

args = parser.parse_args()

# Open the camera
camera = dv.io.CameraCapture()

# Initialize event accumulator with the known resolution
accumulator = dv.Accumulator(camera.getEventResolution())

# ---- Pixel brightness settings ----
# The parameters in this section are expressed in relative values in range [0.0; 1.0], where 0.0 is lowest pixel
# brightness value and 1.0 is the brightest. These values are automatically converted into 8-bit unsigned integer
# representation while generating final image, where the range is [0; 255].

# Event contribution sets a value which an event contributes to pixel brightness
accumulator.setEventContribution(0.25)

# Initial value for pixel brightness, also pixel brightness values are decaying into this value over time
accumulator.setNeutralPotential(0.5)

# Minimum brightness value a pixel can be assigned
accumulator.setMinPotential(0.0)

# Maximum brightness value a pixel can be assigned
accumulator.setMaxPotential(1.0)
# ---- End pixel brightness settings ----

# ---- Decay settings ----
# Decay is an algorithm which applies certain "degradation" of pixel brightness over time. Decay improves
# reconstructed image quality. To read more about the available settings, please refer to DV documentation:
# https://inivation.gitlab.io/dv/dv-docs/docs/accumulator-module/#accumulator-settings-overview

# Decay function selection, possible options: EXPONENTIAL, LINEAR, STEP, NONE
accumulator.setDecayFunction(dv.Accumulator.Decay.LINEAR)

# Decay parameter, this value has different impact depending on selected decay function
accumulator.setDecayParam(1e-6)

# If synchronous decay is enabled, all pixels decay at frame generation. If set to false, the values only get decayed
# when an event is registered at the pixel coordinates.
accumulator.setSynchronousDecay(False)
# ---- End decay settings ----

# Event polarity rectification assumes positive polarity to all events.
accumulator.setRectifyPolarity(False)

# Create the preview window
cv.namedWindow("Preview", cv.WINDOW_NORMAL)


# Event accumulation callback
def accumulate_events(event_slice):
    # Pass event slice into the accumulator
    accumulator.accept(event_slice)

    # Generate a frame
    frame = accumulator.generateFrame()

    # Show a preview of the accumulated frame
    cv.imshow("Preview", frame.image)
    cv.waitKey(2)


# Create an event slicer, this will only be used events only camera
slicer = dv.EventStreamSlicer()

# Perform time-based slicing every 33 milliseconds
slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=33), accumulate_events)

# Start read loop, run while the camera is still connected.
while camera.isConnected():
    # Read events from the camera
    events = camera.getNextEventBatch()

    # If read successful
    if events is not None:
        # Pass the events into a slicer, which takes care of timing and slicing
        slicer.accept(events)
