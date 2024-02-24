
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import datetime

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("01.mp4")
track_history = defaultdict(lambda: [])
def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer
def main():
 writer = create_video_writer(cap, "0000000.mp4")
 while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    start = datetime.datetime.now()
    if success:
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        if boxes is not None:
         for box, track_id in zip(boxes, track_ids):
             x, y, w, h = box
             cv2.putText(frame,str(track_id), (int(x),int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.8, 255)


        end = datetime.datetime.now()
        total = (end - start).total_seconds()
        fps = f"FPS: {1 / total:.2f}"
        writer.write(frame)
        #cv2.imshow("YOLOv8 Tracking",frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
 cap.release()
 cv2.destroyAllWindows()

if __name__=="__main__":
    main()





