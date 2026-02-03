import cv2
from ultralytics import YOLO
from collections import deque
import os

class SmartEye:
    def __init__(self):
        # 1. تحميل المودل الخاص
        model_path = 'best2.pt' 
        
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: '{model_path}' not found. Loading standard yolov8n.pt...")
            self.model = YOLO('yolov8n.pt')
        else:
            print(f"🚀 Loading Custom Model: {model_path}...")
            self.model = YOLO(model_path)

        self.frame_buffer = deque(maxlen=30)
        self.class_config = {}

        # ==============================================================================
        # ⚙️ إعداد القواعد (Mapping Rules)
        # هنا نحدد الإعدادات بناءً على "الاسم" وليس الرقم (أضمن وأدق)
        # ==============================================================================
        
        # 1. قائمة العنف (نطلب ثقة عالية جداً لتجنب الإزعاج)
        high_risk_keywords = ['Violence', 'violence', 'Stabbing', 'Knife_Deploy']
        
        # 2. قائمة الأسلحة النارية (ثقة متوسطة)
        firearms = ['gun', 'handgun', 'automatic rifle', 'shotgun', 'sniper', 'SMG', 
                    'grenade launcher', 'bazooka-', 'Pipe Bombs']
        
        # 3. قائمة الأسلحة البيضاء والحادة
        sharp_weapons = ['knife', 'Knife_Weapon', 'sword', 'Box Cutters', 'shivs', 
                         'Chainsaws', 'Crowbars', 'SAI', 'tONFA', 'brass-knuckle', 'Kusari-fundo']

        # 4. الأشخاص والمجرمين
        # (يمكنك إضافة أي كلاس آخر هنا)

        print("🔧 Configuring Class Rules based on your model...")
        
        # نلف على كل الكلاسات الموجودة داخل المودل ونعطيها الإعدادات المناسبة
        for id, name in self.model.names.items():
            
            # إعدادات افتراضية
            conf = 0.55
            color = (255, 255, 0) # سماوي (للأشياء غير المعروفة)
            is_target = False

            # تصنيف الكلاسات
            if name in high_risk_keywords:
                conf = 0.88     # 🔥 عنف: ثقة عالية
                color = (0, 0, 255) # أحمر
                is_target = True
                
            elif name == 'criminal':
                conf = 0.70
                color = (0, 0, 150) # أحمر غامق
                is_target = True
                
            elif name in firearms:
                conf = 0.60
                color = (0, 0, 255) # أحمر
                is_target = True
                
            elif name in sharp_weapons:
                conf = 0.60
                color = (0, 165, 255) # برتقالي
                is_target = True
                
            elif name == 'person':
                conf = 0.50
                color = (0, 255, 0) # أخضر
                is_target = True

            # إذا كان الكلاس مهماً لنا، نحفظ إعداداته برقمه (ID)
            if is_target:
                self.class_config[id] = {'name': name, 'min_conf': conf, 'color': color}
                print(f"   ✅ Active: ID {id} -> {name} (Min Conf: {conf})")

    def detect_and_draw(self, frame):
        # الكشف
        results = self.model(frame, verbose=False)
        
        detected_objects = []
        box_coordinates = [] 

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # هل الكلاس هذا موجود في قائمتنا المفلترة؟
                if cls_id in self.class_config:
                    
                    config = self.class_config[cls_id]
                    
                    # التحقق من نسبة الثقة الخاصة بهذا الكلاس
                    if conf >= config['min_conf']:
                        
                        label_name = config['name']
                        color = config['color']
                        
                        if label_name not in detected_objects:
                            detected_objects.append(label_name)
                        
                        # الإحداثيات
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label_text = f"{label_name} {conf:.0%}"
                        
                        # تجميع البيانات
                        box_coordinates.append((x1, y1, x2, y2, label_text, color))

                        # الرسم
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # خلفية للنص
                        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
                        
                        cv2.putText(frame, label_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self.frame_buffer.append(frame)
        
        return frame, False, detected_objects, box_coordinates