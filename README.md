# Criminal Detection System

## Overview

This repository contains a comprehensive Criminal Detection System, leveraging cutting-edge technology to remotely register and track criminals through the integration of criminal data. The system offers two distinct identification methods: manual photo input and live webcam recognition.


## Project Structure

```

.
├── criminal_data
│   ├── virat_kohli
|     ├── details.json
|   ├── anushka_sharma
├── known_faces
│   ├── virat_kohli
│     ├── virat_kohli_1.jpg
|     ├── virat_kohli_2.jpg
│   ├── anushka_sharma
├── output_images
│   ├── virat_kohli_rect.jpg
├── static
│   ├── images
├── templates
│   ├── index.html
│   ├── detect.html
│   ├── register.html
│   ├── video.html
├── myapp.py
├── myClass.py
├── trained_knn_model
├── requirements.txt
├── README.md


```

- **criminal_data :** Contains Criminal Data.
- **myClass.py :** Python source code for data preprocessing, feature engineering, model training, and utility functions.
- **trained_knn_model.clf :** Pre-trained machine learning model (KNN).
- **myapp.py :** A  web application for deploying the trained model.
- **README.md :** Documentation for the project.

 ## Getting Started

1. Clone the repository:
 ```
    git clone https://github.com/Ankithmj07/Criminal-Detection-System.git
    cd Criminal-Detection-System
 ```

2. Install the required packages:
```
    pip install -r requirements.txt
```

5.  Explore the web application for model deployment.

## Model Deployment

The pre-trained model is stored in the **'trained_knn_model.clf'** file. You can use this model for predictions without retraining.

## Web Application

A Good web application has been created for deploying the trained model. To run the application, execute the following command:

```
python myapp.py
```

Visit http://localhost:8080 in your web browser to interact with the application.

## Feel free to contribute, report issues, or provide feedback. Happy coding!
