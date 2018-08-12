import cv2
import dlib


class Tracker:

    def __init__(self, tracker_type_index=4):
        self.tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'DLIB']
        self.tracker_type = self.tracker_types[tracker_type_index]

        if self.tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if self.tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if self.tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if self.tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if self.tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if self.tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if self.tracker_type == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()
        if self.tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        if self.tracker_type == 'DLIB':
            self.tracker = dlib.correlation_tracker()

        self.t_x = 0
        self.t_y = 0
        self.t_w = 0
        self.t_h = 0


    def startTrack(self, frame, bbox):
        (self.t_x, self.t_y, self.t_w, self.t_h) = bbox
        # offset = int(0.05*w)
        offset = 0
        expanded_bbox = (self.t_x - offset, 
                        self.t_y - offset, 
                        self.t_x + self.t_w + offset, 
                        self.t_y + self.t_h + offset)

        if self.tracker_type == 'DLIB':
            self.tracker.start_track(frame, dlib.rectangle( expanded_bbox[0],
                                                            expanded_bbox[1],
                                                            expanded_bbox[2],
                                                            expanded_bbox[3]))
        else:
            # Initialize tracker with first frame and bounding box
            # bbox = cv2.selectROI(frame, False)
            expanded_bbox = (self.t_x, self.t_y, self.t_w, self.t_h)
            ok = self.tracker.init(frame, expanded_bbox)


    def update(self, frame):
        # Update tracker
        if self.tracker_type == 'DLIB':
            trackingQuality = self.tracker.update(frame)
            bbox = self.tracker.get_position()
            self.t_x = int(bbox.left())
            self.t_y = int(bbox.top())
            self.t_w = int(bbox.width())
            self.t_h = int(bbox.height())
        else:
            trackingQuality, bbox = self.tracker.update(frame)
            [self.t_x, self.t_y, self.t_w, self.t_h] = bbox

        offset = int(0.0*self.t_w)
        self.t_x -= offset
        self.t_y -= offset
        self.t_w += 2*offset
        self.t_h += 2*offset
        return trackingQuality, [self.t_x, self.t_y, self.t_w, self.t_h]