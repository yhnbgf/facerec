import numpy as np
import face_recognition as fr
import cv2
import os
import time

fps_start_time=0
fps=0


video_capture = cv2.VideoCapture(0)

lucas_image = fr.load_image_file("lucas chin.jpg")
bill_image = fr.load_image_file("megan.jpg")
eugene_image= fr.load_image_file("eugene.jpg")
lucas_face_encoding = fr.face_encodings(lucas_image)[0]
bill_face_encoding = fr.face_encodings(bill_image)[0]
eugene_face_encoding = fr.face_encodings(eugene_image)[0]


known_face_encondings = [bill_face_encoding, lucas_face_encoding, eugene_face_encoding]


known_face_names = ["Megan Chin", "Lucas", "Eugene"]

while True:
    ret, frame = video_capture.read()
    fps_end_time=time.time()
    time_diff=fps_end_time - fps_start_time
    fps=1/(time_diff)
    fps_start_time=fps_end_time
    fps_text= "FPS is {:.2F}".format(fps)  
    cv2.putText(frame,fps_text,(5,30),cv2.FONT_HERSHEY_TRIPLEX , 1, (0,255,255), 1)

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        '''
        if name=="Eugene":
            os.system("say 'Acess granted Eugene', ")
        else:
            os.system("say 'Acess denied', ")
            '''
            




    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

