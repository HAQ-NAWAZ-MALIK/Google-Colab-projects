
[![MasterHead]([https://cdn.dribbble.com/users/1373613/screenshots/5510801/media/b82469d51c432c2ff65c0158334cfabf.gif)

# Face Recognition using OpenCV
![image](https://github.com/HAQ-NAWAZ-MALIK/Google-Colab-projects/assets/86514900/486de2ad-760a-4da2-b83d-b0fba20996e8)

This Python script demonstrates face recognition using OpenCV (Open Source Computer Vision Library). It utilizes the Local Binary Patterns Histograms (LBPH) algorithm for face recognition.


## Prerequisites

- Python 3.x
- OpenCV (cv2) library

## Usage

1. Train the face recognizer with a set of labeled images using a separate script or tool. This will generate a `trainer.yml` file containing the trained model.

2. Update the following paths in the script:
  - `'path to trainer.yml'`: Path to the trained `trainer.yml` file.
  - `'Path of TestImage\\test.jpg'`: Path to the test image for face recognition.

3. Run the script.

4. For image-based face recognition, the script will display the test image with rectangles around the detected faces and labels indicating whether the face is known or unknown.

5. For video-based face recognition, the script will open the default camera (video_capture = cv2.VideoCapture(0)). It will continuously capture frames from the camera, detect faces, and display them in a window with rectangles around the detected faces and labels indicating whether the face is known or unknown.

6. Press the 'q' key to exit the video capture loop.

## How it Works

1. The script loads the pre-trained face recognizer from the `trainer.yml` file.

2. For image-based face recognition:
  - It loads the test image and converts it to grayscale.
  - It detects faces in the grayscale image using the Haar Cascade classifier.
  - For each detected face, it predicts the label and confidence using the loaded face recognizer.
  - It draws a rectangle around the face and displays the predicted label and confidence.

3. For video-based face recognition:
  - It opens the default camera using `cv2.VideoCapture(0)`.
  - In a loop, it captures frames from the camera and converts them to grayscale.
  - It detects faces in the grayscale frame using the Haar Cascade classifier.
  - For each detected face, it predicts the label and confidence using the loaded face recognizer.
  - It draws a rectangle around the face and displays the predicted label and confidence.
  - The loop continues until the 'q' key is pressed.

## Notes

- This script assumes that you have already trained the face recognizer with labeled images. The training process is not included in this script.
- The script uses the LBPH algorithm for face recognition, but you can modify it to use other algorithms available in OpenCV, such as Eigenfaces or Fisherfaces.
- The confidence threshold for determining whether a face is known or unknown is set to 70 in this script. You can adjust this value based on your requirements.

## Resources

- OpenCV Documentation: https://docs.opencv.org/
- OpenCV Face Recognition: https://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html
