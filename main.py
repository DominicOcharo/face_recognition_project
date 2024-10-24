from fastapi import FastAPI, HTTPException
import cv2
import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from threading import Thread
import time

app = FastAPI()

data_folder = "data"
haar_cascade_path = os.path.join(data_folder, "haarcascade_frontalface_default.xml")
facedetect = cv2.CascadeClassifier(haar_cascade_path)

def capture_faces(name: str):
    video = cv2.VideoCapture(0)
    faces_data = []
    i = 0

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) == 100:
            break

    video.release()
    cv2.destroyAllWindows()

    # Save the captured faces
    faces_data = np.asarray(faces_data).reshape(100, -1)

    if not os.path.exists(os.path.join(data_folder, 'names.pkl')):
        names = [name] * 100
        with open(os.path.join(data_folder, 'names.pkl'), 'wb') as f:
            pickle.dump(names, f)
    else:
        with open(os.path.join(data_folder, 'names.pkl'), 'rb') as f:
            names = pickle.load(f)
        names += [name] * 100
        with open(os.path.join(data_folder, 'names.pkl'), 'wb') as f:
            pickle.dump(names, f)

    if not os.path.exists(os.path.join(data_folder, 'faces_data.pkl')):
        with open(os.path.join(data_folder, 'faces_data.pkl'), 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open(os.path.join(data_folder, 'faces_data.pkl'), 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open(os.path.join(data_folder, 'faces_data.pkl'), 'wb') as f:
            pickle.dump(faces, f)

@app.post("/register")
def register_user(name: str):
    """
    Register a new user by capturing their face data for 100 seconds.
    """
    capture_thread = Thread(target=capture_faces, args=(name,))
    capture_thread.start()
    return {"message": f"Registration in progress for {name}. Please wait for 100 seconds and then check the saved data."}

def identify_user():
    video = cv2.VideoCapture(0)
    
    with open(os.path.join(data_folder, 'names.pkl'), 'rb') as f:
        LABELS = pickle.load(f)

    with open(os.path.join(data_folder, 'faces_data.pkl'), 'rb') as f:
        FACES = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    start_time = time.time()

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            video.release()
            cv2.destroyAllWindows()
            return output[0]

        if time.time() - start_time > 15:
            break

    video.release()
    cv2.destroyAllWindows()
    return None

@app.post("/login")
def login_user():
    """
    Login a user by identifying their face within 15 seconds.
    If the model's prediction is uncertain (i.e., distance too high), reject the login.
    """
    video = cv2.VideoCapture(0)

    # Load the known face data and labels
    with open(os.path.join(data_folder, 'names.pkl'), 'rb') as f:
        LABELS = pickle.load(f)

    with open(os.path.join(data_folder, 'faces_data.pkl'), 'rb') as f:
        FACES = pickle.load(f)

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    start_time = time.time()

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        # For every detected face in the frame
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

            # Get the distance to the nearest neighbors
            distances, indices = knn.kneighbors(resized_img, n_neighbors=1)
            closest_distance = distances[0][0]  # Smallest distance

            # Set an appropriate threshold based on experimentation
            threshold = 0.6  # Adjust this threshold depending on your data
            
            if closest_distance < threshold:
                predicted_label = knn.predict(resized_img)[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 0, 255), -1)
                cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

                # Show the frame with the face and predicted name
                cv2.imshow("Login - Face Recognition", frame)

                # If a face is identified with confidence, close the video feed and return the result
                video.release()
                cv2.destroyAllWindows()
                return {"message": f"Login successful. Welcome, {predicted_label}!"}
            else:
                # If the face is not confidently recognized, mark it as unknown
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
                cv2.imshow("Login - Face Recognition", frame)

        # Show the frame even if no face is detected yet
        cv2.imshow("Login - Face Recognition", frame)

        # Stop scanning after 15 seconds
        if time.time() - start_time > 15:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    raise HTTPException(status_code=401, detail="Login unsuccessful. Please register or try again.")
