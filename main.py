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


# python main.py -i videos/example_01.mp4


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="minimum probability to filter weak detections")

    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(
        "model/MobileNetSSD_deploy.prototxt", "model/MobileNetSSD_deploy.caffemodel")
    if not args.get("input", False):
        print("[STATUS] Starting the live stream..")
        vs = VideoStream(config.url).start()
        time.sleep(2.0)
    else:
        print("[STATUS] Video Running..")
        vs = cv2.VideoCapture(args["input"])
    writer = None

    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    totalfps = 0
    totalEnter = 0
    totalExit = 0

    traceableObjects = {}
    tracer = []
    x = []
    exit = []
    enter = []
    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    while True:
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        if args["input"] is not None and frame is None:
            break

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (W, H), True)

        status = "Waiting"
        rects = []

        if totalfps % 30 == 0:

            status = "Detecting"
            tracer = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > args["confidence"]:

                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    tracer.append(tracker)

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

        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 255), 3)
        # cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():

            to = traceableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)

            else:

                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:

                    if direction < 0 and centroid[1] < H // 2:
                        totalExit += 1
                        exit.append(totalExit)
                        to.counted = True

                    elif direction > 0 and centroid[1] > H // 2:
                        totalEnter += 1
                        enter.append(totalEnter)

                        x = []

                        x.append(len(enter)-len(exit))

                        if sum(x) > config.Threshold:

                            if config.ALERT:
                                cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
                                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                                print("[STATUS] Playing Sound..")
                                Sound().play()
                                print("[STATUS] Alerted..")

                        to.counted = True

            traceableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(
                frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        info = [
            ("Exit", totalExit),
            ("Enter", totalEnter),
            ("Status", status),
        ]

        info2 = [
            ("People Inside", x),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (224, 54, 105), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (265, H - ((i * 20) + 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Crowd Control System", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        totalfps += 1
        fps.update()

    fps.stop()

    cv2.destroyAllWindows()


main()
