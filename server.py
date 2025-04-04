from flask import Flask, Response, jsonify,request
from flask_socketio import SocketIO #Unlike normal HTTP requests, WebSockets keep a persistent connection open
from flask_cors import CORS
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import base64


app = Flask(__name__) # Instance of flask app 
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

socketio = SocketIO(app, cors_allowed_origins="*") # Allow real-time communication in app

model_dict = pickle.load(open('./model.p','rb')) 
# print(model_dict.keys())

model = model_dict['model']

# Camera iniitally off
cap = None 

# Tools to detect and draw landmarks in images 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = labels_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
    26: 'Z',
}

'''
Reads webcam frames.
Detects hands.
Predicts ASL characters.
Yields frames 
'''


@app.route('/predict', methods=['POST'])
def predict():
    predicted_character = ''
    expected_features = model.n_features_in_

    data = request.get_json()
    image_data = data['image'].split(',')[1]  # Strip header
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    data_aux = []
    x_ = []
    y_= []
    H,W,_ = frame.shape
        
    # Converts frame to rgb format 
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  

    # Detect landmarks (hands) in image
    results = hands.process(frame_rgb) 
    
    
    # Landmark coordinates lie in results.multi_hand_landmarks 
    if results.multi_hand_landmarks:
        predicted_character = ''
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw hand landmark on frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            # Extract hand landmark coordinates
            for coord in  hand_landmarks.landmark:
                x = coord.x
                y = coord.y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
        #print("Feature count:", len(data_aux))  # Should be 42
        #print("Expected:", model.n_features_in_)  # Should match 42

        
        # Inputs hand coords into model & Returns predicted character to frontend 
        try:
                # Check if feature count matches model expectations
                if len(data_aux) != expected_features:
                    raise ValueError(f"Feature mismatch: Expected {expected_features}, but got {len(data_aux)}")
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                print('PREDICTED CHARACHTER: ',predicted_character)
                    # If 2 second has passed since the last prediction was sent
                #current_time = time.time()
                
                # if current_time - last_sent_time >= 2:
                #     last_sent_time = current_time  # Update last sent timestamp

                #     # Send prediction to frontend
                #     socketio.emit("prediction", {"character": predicted_character})

        except ValueError as e:
            #print(f"⚠️ Prediction Error: {e}")  # Log error in Flask console
            #socketio.emit("error", {"message": str(e)})  # Send error to frontend
            pass

        
        # Calculates bounding box around hand landmark 
        x1,y1= int(min(x_) * W),int(min(y_) * H)   # smallest x value in landmark coords scaled to width of frame 
        x2,y2= int(max(x_) * W), int(max(y_) * H)
        
        #Draw rectangle around landmark 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        
        # Add text above rectangle
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Run prediction here (use your model and logic)
    #predicted_character = "A"  # Example placeholder

    return jsonify({'character': predicted_character})

if __name__ == '__main__':
    socketio.run(app)
    #socketio.run(app, host='0.0.0.0', port=5000,allow_unsafe_werkzeug=True)





