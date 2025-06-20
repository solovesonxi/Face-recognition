# Face Recognition System Implementation Report

## Members

- 陈世有 12211931 25%
- 文启豪 12112119 25%
- 徐进哲 12211614 25%
- 朱炫睿 12211930 25%

## 1 Introduction

This project presents a comprehensive face recognition system developed using Python, designed to provide advanced facial analysis capabilities through a user-friendly graphical interface. The system leverages cutting-edge deep learning models via the DeepFace library, the primary goal of this system is to offer multiple face recognition functionalities in a single application, including face verification, face identification, facial attribute analysis, and real-time face recognition from a live camera feed. The application is designed to be accessible to both technical and non-technical users, with a focus on usability and visual appeal.

## 2 Related Work

### 2.1 RetinaFace for Face Detection

RetinaFace is a single-stage face detection model that stands out for its ability to achieve multi-level face localization. The model incorporates a feature pyramid network (FPN) to effectively capture faces of various scales, ensuring robust performance even for tiny faces that are often challenging to detect. What’s more, RetinaFace introduces a novel anchor design and a context-aware module to enhance the detection of faces in complex scenes with heavy occlusions or dense crowds.

![1750410890616](image/Report/1750410890616.png)

### 2.2 Facenet for Face Recongnition

FaceNet is a cutting-edge deep learning model designed to transform faces into compact and discriminative embeddings, enabling efficient face recognition, verification, and clustering. At its core, it uses a convolutional neural network (CNN) architecture to map facial images into a high-dimensional feature space where the Euclidean distance between embeddings directly correlates with the similarity of faces. The model is trained with triplet loss, a strategy that pushes embeddings of the same face (anchor and positive) closer together while pulling embeddings of different faces (negative) apart, thereby enhancing the discriminative power of the learned representations.

![1750410879407](image/Report/1750410879407.png)

### 2.3 DeepFace Library: Architecture and Functionality

DeepFace is an open-source Python library for face analysis. It abstracts complex deep learning models into simple API calls, supporting face detection, alignment and attribute analysis. DeepFace handles all the face recognition pipeline in the background, users do not need to acquire in-depth knowledge about all the processes behind it and  can just call its verification, find or analysis function with a single line of code.

### 2.4 VGG-Face2 Dataset: Structure and Significance

VGG-Face2 is a large-scale face recognition dataset developed by the Visual Geometry Group (VGG) at Oxford University. It extends the original VGG-Face dataset with 3.31 million images of 9,131 identities (8,631 training, 500 validation). This dataset captures variations in pose, age, ethnicity, and occlusion. All the images are collected from YouTube videos, covering a wide range of real-world scenarios.

VGG-Face2 is structured as follows (CSV files map image paths to identities, with attributes like gender and age group):

```
vggface2/
├── train/
│   ├── identity_1/
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── ...
│   ├── identity_2/
│   └── ...
├── val/
│   ├── identity_8632/
│   └── ...
└── meta/
    ├── train.csv
    └── val.csv
```

## 3 Approach

The face recognition system is structured into several key components that work together to provide the desired functionality:

- **Tkinter:** The standard GUI library for Python, used to build the application interface
- **OpenCV (cv2):** For camera access and basic image processing
- **PIL (Pillow):** For advanced image manipulation and display within the GUI
- **DeepFace:** The core library for face recognition, verification, and attribute analysis
- **Threading:** For implementing multi-threading to ensure the UI remains responsive during intensive facial analysis

### 3.1 GUI Initialization

The `__init__` method of the FaceRecognitionApp class sets up the main window with a specific title, size, and background color. It binds the <Escape> key to stop the real - time analysis. An icon is attempted to be set for the application. The GUI has a top title bar, a sidebar for navigation with function buttons, and a main area divided into an image preview tab and an analysis results tab. The sidebar also contains a status label to show the current state of the application.

### 3.2 Face Verification

The `show_verify_screen` method is responsible for face verification. It prompts the user to select two images. If both images are selected, it uses the `DeepFace.verify` function to check if the faces in the two images match. The result, including verification status, similarity, distance, and the model used, is displayed in the analysis results tab. If an error occurs during verification, an error message is shown, and details are logged in the output area.

### 3.3 Face Recognition

The `show_find_screen` method enables face recognition. It asks the user to select an image to be recognized and a database folder. After selection, it uses the `DeepFace.find` function to search for matching faces in the database. The top 5 matching results, including identity, similarity, and file path, are displayed in the analysis results tab. If no match is found, a corresponding message is shown.

### 3.4 Facial Attribute Analysis

The `show_analyze_screen` method performs facial attribute analysis. It requests the user to choose an image. Once selected, it uses the `DeepFace.analyze` function to extract attributes such as age, gender, race, and emotion. The results are formatted and displayed in the analysis results tab, with color-coded gender results based on confidence levels. Detailed probability data for gender, race, and emotion are also provided.

### 3.5 Realtime Analysis

The realtime analysis functionality is managed by several methods. The `toggle_stream_analysis` method toggles between starting and stopping the realtime analysis. The `start_stream_analysis` method asks the user to select a database folder. After selection, it starts a new thread to perform the realtime analysis using the `stream_analysis` method. The `stream_analysis` method opens the webcam, captures frames, and uses `DeepFace.find` to recognize faces in the frames. If a match is found, the identity and similarity are displayed on the frame. The analysis can be stopped by pressing the <ESC> key, which calls the stop_stream_analysis method. This method releases the webcam resources, closes the OpenCV window, and restores the realtime analysis button to its original state.

## 4 Experimental Results

## 5 Conclusion

## 6 References

