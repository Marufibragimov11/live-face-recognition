import cv2
import face_recognition

"""
Link to download other pre-trained models
https://github.com/opencv/opencv/tree/master/data/haarcascades
"""

# Load the pre-trained face recognition model
model = "data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(model)

# Known faces data (replace with your images and names)
known_images = ["data/my_photo.jpg", "data/smiling_man.jpg"]
known_names = ["Maruf", "Steve"]

known_encodings = []
for image in known_images:
    # Load the image
    img = cv2.imread(image)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = face_recognition.face_locations(rgb_img)

    # Get face encodings
    for face in faces:
        encoding = face_recognition.face_encodings(rgb_img, [face])[0]
        known_encodings.append(encoding)

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all faces in the frame
    faces = face_cascade.detectMultiScale(rgb_frame, 1.1, 4)

    # Recognize faces
    for (x, y, w, h) in faces:
        # Extract the face ROI
        roi = rgb_frame[y:y + h, x:x + w]

        # Check if ROI has a detected face
        face_locations = face_recognition.face_locations(roi)
        if len(face_locations) == 0:
            continue  # Skip to the next face (or frame if no more faces)

        # Encode the face
        encoding = face_recognition.face_encodings(roi)[0]

        # Compare the face encoding with known encodings
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Draw a bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Add a label with the name below the face
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Quit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
video_capture.release()
cv2.destroyAllWindows()
