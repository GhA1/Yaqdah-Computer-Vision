from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch
import cv2

class ActionRecognizer:
    def __init__(self):
        print("Loading VideoMAE Model...")
        model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
        
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
            self.model = VideoMAEForVideoClassification.from_pretrained(model_ckpt).to(self.device)
            print(f"✅ VideoMAE Loaded on {self.device}.")
        except Exception as e:
            print(f"❌ Error: {e}")
            self.model = None

        # ====================================================
        # ⚠️ قاموس الخطر (تم التحديث بناءً على السجلات)
        # ====================================================
# أرجع القائمة نظيفة (احذف motorcycling و driving و skateboarding)
        self.danger_keywords = [
            'fighting', 'punching', 'kicking', 'slapping', 'brawling', 
            'stabbing', 'hitting', 'headbutting', 'wrestling', 'choking',
            'drop kicking', 'shooting', 'sword', 'gun', 'knife',
            'weapon', 'pushing car' # أبقِ parkour إذا كنت تعتبرها خطرة
        ]

    def predict(self, frame_buffer):
        if self.model is None or len(frame_buffer) < 16:
            return "Buffering...", False, 0.0

        try:
            # تجهيز الفريمات
            indices = np.linspace(0, len(frame_buffer) - 1, 16).astype(int)
            clip = [frame_buffer[i] for i in indices]
            clip_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in clip]

            inputs = self.processor(list(clip_rgb), return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # ==========================================================
            # 🔥 التغيير الجوهري: فحص أعلى 5 نتائج بدلاً من الأولى فقط
            # ==========================================================
            top5_prob, top5_indices = torch.topk(probs, 5)
            
            final_label = "Normal"
            final_conf = 0.0
            is_danger = False
            
            print("\n🔍 AI Scan (Top 5):")
            
            for i in range(5):
                label_text = self.model.config.id2label[top5_indices[0][i].item()]
                score = top5_prob[0][i].item() * 100
                print(f"   {i+1}. {label_text}: {score:.1f}%")
                
                # التحقق هل هذه الكلمة خطرة؟
                if any(k in label_text.lower() for k in self.danger_keywords):
                    is_danger = True
                    final_label = label_text # نأخذ اسم الخطر حتى لو كان ترتيبه الثالث
                    final_conf = score       # نأخذ نسبته
                    print(f"      >>> ⚠️ DETECTED THREAT: {label_text}")
                    break # وجدنا خطر، نتوقف عن البحث ونبلغ فوراً

            # إذا لم نجد خطراً في الـ 5 الأوائل، نأخذ النتيجة رقم 1 كالعادة
            if not is_danger:
                final_label = self.model.config.id2label[top5_indices[0][0].item()]
                final_conf = top5_prob[0][0].item() * 100

            # توحيد المسمى للعرض
            display_label = "Violence Detected" if is_danger else final_label

            return display_label, is_danger, final_conf

        except Exception as e:
            print(f"Error: {e}")
            return "Error", False, 0.0