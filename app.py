from flask import Flask, render_template, request
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import torch
import base64
import io
from datetime import datetime

app = Flask(__name__)
# ✅ Increase file upload size limit to 16 MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def save_base64_image(base64_str, upload_folder):
    header, encoded = base64_str.split(',', 1)
    data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    filename = f"cropped_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
    filepath = os.path.join(upload_folder, filename)
    img.save(filepath)
    return filepath

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    image_path = None

    if request.method == 'POST':
        if request.form.get('croppedImage'):  # If using Cropper.js
            base64_data = request.form['croppedImage']
            filepath = save_base64_image(base64_data, app.config['UPLOAD_FOLDER'])
            image_path = filepath
        elif 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{image.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(filepath)
                image_path = filepath

        # Generate caption
        if image_path:
            img = Image.open(image_path).convert('RGB')
            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

    return render_template('index.html', caption=caption, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
