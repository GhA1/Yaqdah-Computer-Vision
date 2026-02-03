import tensorflow as tf
import cv2
import numpy as np

class KerasClassifier:
    def __init__(self, model_path='my_model.keras'):
        print(f"Loading Keras Model from {model_path}...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Keras Model Loaded Successfully.")
        except Exception as e:
            print(f"❌ Error loading Keras model: {e}")
            self.model = None

        # ========================================================
        # ⚠️ هام جداً: تأكد أن ترتيب هذه القائمة يطابق ترتيب مجلدات التدريب أبجدياً
        # ========================================================
        self.class_names = [
            'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
            'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 
            'Robbery', 'Shooting', 'Shoplifting', 'Snatch', 
            'Stealing', 'Vandalism'
        ]
        # ملاحظة: إذا كان عدد الكلاسات عندك 14 بالضبط، فربما أحد الكلاسات أعلاه غير موجود
        # أو أن Normal تحل محل واحد منهم. (عدل القائمة لتطابق مجلداتك بالضبط)

        # حجم الصورة
        self.img_height = 64
        self.img_width = 64
        
        # قائمة "الأمان"
        self.safe_classes = ['Normal'] 

    def predict(self, frame):
        if self.model is None:
            return "Error", False, 0.0

        try:
            # تجهيز الصورة
            img = cv2.resize(frame, (self.img_width, self.img_height))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # (اختياري: جرب تفعيل أو تعطيل القسمة حسب دقة النتائج)
            # img_array = img_array / 255.0

            # التنبؤ
            predictions = self.model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])

            class_index = np.argmax(score)
            confidence = 100 * np.max(score)
            predicted_class = self.class_names[class_index]

            # ====================================================
            # منطق الخطر الجديد
            # ====================================================
            # إذا كان الكلاس ليس "Normal" + والثقة عالية > 75%
            is_danger = (predicted_class not in self.safe_classes and confidence > 75)

            return predicted_class, is_danger, confidence

        except Exception as e:
            print(f"Prediction Error: {e}")
            return "Error", False, 0.0