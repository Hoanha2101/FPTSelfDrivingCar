from flask import Flask, request, jsonify
import cv2
import numpy as np
from utils import pipeline_function 
from config import *

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'error': 'Invalid image format'}), 400
    
    img_, direction_return, angle_return = pipeline_function(frame, INTEREST_BOX, paint=True, lane_paint=True, interest_box=True)
    
    # Chuyển đổi img_ thành dữ liệu có thể gửi đi
    _, img_encoded = cv2.imencode('.jpg', img_)
    img_bytes = img_encoded.tobytes()
    
    return jsonify({
        'direction_return': direction_return,
        'angle_return': angle_return,
        'processed_image': img_bytes.hex()  # Encode ảnh thành hex string để gửi qua JSON
    })

if __name__ == '__main__':
    app.run(debug=True)
