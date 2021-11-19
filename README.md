- keypoint classifier

Flows:
1. read video (frames)
2. Loop frame (image)

image -> mediapipe => get keypoints -> pass all keypoints to keypoint classifier -> get result (predicted output) -> display text on video/image/frame