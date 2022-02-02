from modules.centroidtracker import CentroidTracker
from modules.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from modules import config, thread
from sound import Sound
import time
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2



def main():
    #Build the argument and parse it
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="minimum probability to filter weak detections")

    args = vars(ap.parse_args())
    # Initiate the class labels 
    # MobileNet SSD has been trained with
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    #load the model
    net = cv2.dnn.readNetFromCaffe(
        "model/MobileNetSSD_deploy.prototxt", "model/MobileNetSSD_deploy.caffemodel")
    # if no video input file then take feed from ip camera live stream
    if not args.get("input", False):
        print("[STATUS] Starting the live stream..")
        vs = VideoStream(config.url).start()
        time.sleep(2.0)
    else:
        print("[STATUS] Video Running..")
        vs = cv2.VideoCapture(args["input"])
    writer = None

    #frame dimensions
    W = None
    H = None

    # Initiate our centroid tracker, 
    # then create a list of each of our
    # dlib correlation trackers, 
    # and then create a dictionary to map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    totalfps = 0
    totalEnter = 0
    totalExit = 0

    traceableObjects = {}
    tracer = []
    x = []
    exit = []
    enter = []
    # frame per second for video stream
    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    # loop over each frame
    def_color=(0,255,0)
    while True:
        # grab the next frame and handle 
        # if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        # if we don't grab a frame while viewing a video, we have reached the end of the video
        if args["input"] is not None and frame is None:
            break

        # make frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are not set, then set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (W, H), True)

        # Set up our current status along with a list 
        # of bounding box rectangles returned either by (1) 
        # our object detector or (2) the correlation trackers
        status = "Waiting"
        rects = []

        if totalfps % 30 == 0:
            # Initiate our new set of object trackers and set the status
            status = "Detecting"
            tracer = []

            # The frame is converted to a blob 
            # and is sent through the
            # network to obtain detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # calculate the confidence  associated with the prediction
                confidence = detections[0, 0, i, 2]

                # ignore weak predictions
                if confidence > args["confidence"]:
                    # From the detections list, extract the class label index
                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    # calculate the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Using the bounding box coordinates, 
                    # construct a dlib rectangle object, 
                    # and then start the dlib correlation tracker

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    tracer.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain 
        # a higher frame processing throughput
        else:

            for tracker in tracer:

                status = "Tracking"

                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        # We can determine whether an object is moving 
        # 'up' or 'down' by drawing a horizontal line at the center of the frame
        cv2.line(frame, (0, H // 2), (W, H // 2), def_color, 3)

        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():

            to = traceableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # Alternatively, we can utilize a trackable object to determine direction
            else:

                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    # The direction should be negative (indicating the object is moving upward) 
                    # AND the centroid should be above the center line.
                    if direction < 0 and centroid[1] < H // 2:
                        totalExit += 1
                        exit.append(totalExit)
                        to.counted = True

                    elif direction > 0 and centroid[1] > H // 2:
                        totalEnter += 1
                        enter.append(totalEnter)

                        x = []

                        x.append(len(enter)-len(exit))
                        # if the people at place exceeds over threshold, alert sound is played
                        if sum(x) > config.Threshold:
                            def_color=(0,0,255)

                            if config.ALERT:
                                print("[STATUS] Playing Sound..")
                                Sound().play()
                                print("[STATUS] Alerted..")
                        else:
                            def_color=(0,255,0)
                        to.counted = True

            traceableObjects[objectID] = to

            # Display the ID of the object and its centroid in the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(
                frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # construct a tuple
        info = [
            ("Exit", totalExit),
            ("Enter", totalEnter),
            ("Status", status),
        ]

        info2 = [
            ("People Inside", x),
        ]

        # display the req output
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (224, 54, 105), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (265, H - ((i * 20) + 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # cv2.line(frame, (0, H // 2), (W, H // 2), (0,255,0), 3)


        # Opencv output frame
        cv2.imshow("Crowd Control System", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break out of loop
        if key == ord("q"):
            break

        # count the frames
        totalfps += 1
        fps.update()

    fps.stop()

    cv2.destroyAllWindows()


main()
