import os
from pathlib import Path
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from model import image_tensor, load_model

def create_app(model_path=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8*1024*1024
    default_path = next((candidate for candidate in (
        'artifacts/gold_engineered.pt', 'artifacts/gold_mobilenet.keras',
        'artifacts/gold_refined.pt', 'artifacts/gold.pt')
        if Path(candidate).is_file()), 'artifacts/gold.pt')
    path = Path(model_path or os.environ.get('GOLD_MODEL_PATH', default_path))
    model = None
    is_mobilenet = path.suffix == '.keras'
    if path.is_file():
        if is_mobilenet:
            from mobilenet import keras
            model = keras.models.load_model(path, compile=False)
        else:
            model = load_model(path)
    @app.get('/')
    def index():
        return render_template('index.html', ready=model is not None)
    @app.get('/health')
    def health():
        return jsonify(status='ok', model_ready=model is not None,
                       checkpoint=path.name if model is not None else None,
                       architecture=('MobileNetV2' if is_mobilenet else 'GoldCNN') if model is not None else None)
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error='حجم الملف يتجاوز 8 ميجابايت'), 413
    @app.post('/predict')
    def predict():
        if model is None:
            return jsonify(error='النموذج غير مدرّب. أضف البيانات ودرّبه ثم أعد تشغيل الخادم.'), 503
        upload = request.files.get('image')
        if upload is None or not upload.filename:
            return jsonify(error='اختر صورة أولاً'), 400
        try:
            with Image.open(upload.stream) as image:
                if image.format not in {'JPEG', 'PNG', 'WEBP'} or image.width*image.height > 20_000_000:
                    raise ValueError()
                if is_mobilenet:
                    from mobilenet import image_array
                    tensor = image_array(image)[None]
                else:
                    tensor = image_tensor(image, getattr(model, 'preprocessing', None)).unsqueeze(0)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error='استخدم صورة JPG أو PNG أو WebP سليمة حتى 20 مليون بكسل'), 400
        if is_mobilenet:
            from mobilenet import keras
            with torch.inference_mode():
                scores = keras.ops.convert_to_numpy(model(tensor, training=False))[0]
        else:
            with torch.inference_mode():
                scores = model(tensor).softmax(1)[0]
        return jsonify(label='حقيقي' if scores.argmax().item() == 1 else 'مزيف',
                       real_score=round(scores[1].item()*100, 2), fake_score=round(scores[0].item()*100, 2),
                       warning='تقدير بصري غير معاير وليس إثباتاً لأصالة الذهب أو نقائه.')
    return app

if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=5000, debug=False)
