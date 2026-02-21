import cv2 as cv

DETECTOR_OPTIONS = ['opencv', 'dlib', 'mediapipe', 'mtcnn', 'retinaface', 'yolo']
VALID_EXTS = ('.jpg', '.jpeg', '.png')

MIN_CONFIDENCE = 0.3
DEVICE = 'cpu'

YOLO_MODEL_PATH = "" #path to pretrained YOLO model weights
MP_MODEL_PATH = "" #path to pretrained MediaPipe model weights
# MP_MODEL_SELECTION = 0 #0 -> short-range model, 1 -> full-range model # needed when using mp.solutions, not mp.tasks
OPENCV_HAAR_CASCADES = "" #path to file 'haarcascade_frontalface_default.xml'

class OpenCVDetector:
    def __init__(self):
        self.detector = cv.CascadeClassifier()
        self.detector.load(OPENCV_HAAR_CASCADES)

    def detect(self, img_path):
        image = cv.imread(str(img_path))
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        dets_and_scores = []
        for f in faces:
            x1 = int(f[0])
            y1 = int(f[1])
            x2 = int(x1 + f[2])
            y2 = int(y1 + f[3])
            dets_and_scores.append([x1, y1, x2, y2, -1.0]) #-1.0 because the classifier doesn't provide confidence scores

        return dets_and_scores
    
class DlibDetector:
    def __init__(self):
        import dlib
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, img_path):
        img = cv.imread(str(img_path))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        detections, scores, _ = self.detector.run(img_rgb, 1)

        dets_and_scores = []
        for i in range(len(detections)):
            box = detections[i]
            x1 = int(box.left())
            y1 = int(box.top())
            x2 = int(box.right())
            y2 = int(box.bottom())
            dets_and_scores.append([x1, y1, x2, y2, float(scores[i])])

        return dets_and_scores

class MediaPipeDetector:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = vision.FaceDetector
        FaceDetectorOptions = vision.FaceDetectorOptions
        VisionRunningMode = vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE)
        self.detector = FaceDetector.create_from_options(options)
        self.mp = mp  #reference needed in detect method for mp.Image

        # # version using mp.solutions:
        # mp_face = mp.solutions.face_detection
        # self.detector = mp_face.FaceDetection(model_selection=MP_MODEL_SELECTION,min_detection_confidence=MIN_CONFIDENCE)

    def detect(self, img_path):
        mp_image = self.mp.Image.create_from_file(str(img_path))
        
        result = self.detector.detect(mp_image)

        dets_and_scores = []
        if result.detections:
            for detection in result.detections:
                bbox = detection.bounding_box
                x1 = int(bbox.origin_x)
                y1 = int(bbox.origin_y)
                x2 = int(bbox.origin_x + bbox.width)
                y2 = int(bbox.origin_y + bbox.height)
                score = float(detection.categories[0].score)
                dets_and_scores.append([x1, y1, x2, y2, score])

        return dets_and_scores
    
        # # version using mp.solutions:
        # img = cv.imread(str(img_path))
        # img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # result = self.detector.process(img_rgb)
        # detections = result.detections

        # dets_and_scores = []
        # if detections:
        #     h, w, _ = img.shape
        #     for face in detections:
        #         box = face.location_data.relative_bounding_box
        #         x1 = int(box.xmin * w)
        #         y1 = int(box.ymin * h)
        #         x2 = int((box.xmin + box.width) * w)
        #         y2 = int((box.ymin + box.height) * h)
        #         dets_and_scores.append([x1, y1, x2, y2, float(face.score[0])])
        # return dets_and_scores

    def close(self):
        self.detector.close()


class MTCNNDetector:
    def __init__(self):
        from facenet_pytorch import MTCNN
        self.detector = MTCNN(keep_all=True, device=DEVICE)

    def detect(self, img_path):
        img = cv.imread(str(img_path))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        detections, scores = self.detector.detect(img_rgb)

        dets_and_scores = []
        if detections is not None:
            for i, box in enumerate(detections):
                x1, y1, x2, y2 = map(int, box.tolist())
                dets_and_scores.append([x1, y1, x2, y2, float(scores[i])])

        return dets_and_scores

class RetinaFaceDetector:
    def __init__(self):
        from retinaface import RetinaFace
        self.detector = RetinaFace

    def detect(self, img_path):
        detections = self.detector.detect_faces(str(img_path),threshold=MIN_CONFIDENCE)

        if not detections:
            return []

        dets_and_scores = []
        for key in detections:
            info = detections[key]
            box = info["facial_area"]
            x1, y1, x2, y2 = map(int, box)
            dets_and_scores.append([x1, y1, x2, y2, float(info.get("score", -1.0))])

        return dets_and_scores
    
class YOLODetector:
    def __init__(self):
        from ultralytics import YOLO
        self.detector = YOLO(YOLO_MODEL_PATH)

    def detect(self, img_path):
        detections = self.detector(str(img_path),device=DEVICE,conf=MIN_CONFIDENCE,imgsz=640)

        res = detections[0]

        dets_and_scores = []
        if hasattr(res, "boxes") and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for i, (x1f, y1f, x2f, y2f) in enumerate(boxes):
                x1, y1, x2, y2 = map(int, (x1f, y1f, x2f, y2f))
                dets_and_scores.append([x1, y1, x2, y2, float(confs[i])])

        return dets_and_scores
    
def create_detector(detector_name):
    if detector_name not in DETECTOR_OPTIONS:
        raise ValueError(f"Detector name '{detector_name}' is not valid. Choose from {DETECTOR_OPTIONS}.")
    
    if detector_name == "dlib":
        return DlibDetector()
    elif detector_name == "mediapipe":
        return MediaPipeDetector()
    elif detector_name == "mtcnn":
        return MTCNNDetector()
    elif detector_name == "retinaface":
        return RetinaFaceDetector()
    elif detector_name == "yolo":
        return YOLODetector()
    elif detector_name == "cv":
        return OpenCVDetector()

# useful methods for handling bboxes:
def clip_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2

def add_padding_to_bbox(x1, y1, x2, y2, pad_ratio=0.05):
    padding_x = int(pad_ratio * (x2 - x1))
    padding_y = int(pad_ratio * (y2 - y1))
    return (x1-padding_x, y1-padding_y, x2+padding_x, y2+padding_y)