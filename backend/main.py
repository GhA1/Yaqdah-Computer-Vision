from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import time
import json
import threading
import os
import uuid
import numpy as np # ضروري لدمج الصور
from ai_engine import SmartEye          
from database import save_alert, get_all_alerts
from gemini_agent import generate_security_report 

app = FastAPI(title="Yaqza Smart Guard - Context Aware Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("alerts_images", exist_ok=True)
app.mount("/images", StaticFiles(directory="alerts_images"), name="images")

# ==========================================
# ⚡ إعدادات النظام
# ==========================================
VIDEO_SOURCE = "test_video.mp4" 
RESIZE_DIM = (480, 360)         
FRAME_SKIP_YOLO = 5             

print("🚀 Initializing YOLO Engine...")
yolo_eye = SmartEye() # تأكد أن maxlen في ai_engine هو 300 على الأقل
print("✅ AI Engine Ready.")

current_status = {
    "boxes": [],           
    "danger": False,
    "danger_type": "None"
}

gemini_state = {
    "is_analyzing": False,
    "last_run": 0,
    "cooldown": 60 
}

lock = threading.Lock()

# ==========================================
# 🧠 عامل جيمني (تحليل السياق الزمني ودمج الصور)
# ==========================================
def gemini_worker(frame_buffer_snapshot, trigger_frame, detected_action):
    global gemini_state
    
    print(f"🤖 Processing Context (30s) for: {detected_action}...")
    
    # 1. سحب عينات زمنية (Temporal Sampling)
    sampled_context = []
    if len(frame_buffer_snapshot) > 10:
        # نسحب 5 فريمات من الماضي (تغطي الـ 30 ثانية)
        step = len(frame_buffer_snapshot) // 5
        for i in range(0, len(frame_buffer_snapshot), step):
            sampled_context.append(frame_buffer_snapshot[i])
    
    # القائمة النهائية للتحليل (صور الماضي + صورة الحدث الحالية)
    analysis_images = sampled_context + [trigger_frame]
    
    # 2. إنشاء شريط الأحداث (Filmstrip) للداشبورد
    # نقوم بتصغير الصور ودمجها أفقياً لرؤية تسلسل الحدث
    thumbs = [cv2.resize(img, (160, 120)) for img in analysis_images]
    filmstrip = np.hstack(thumbs) # دمج الصور بجانب بعضها
    
    # 3. إرسال الصور لجيمني
    json_response = generate_security_report(analysis_images, detected_action)
    
    try:
        clean_text = json_response.replace("```json", "").replace("```", "").strip()
        report = json.loads(clean_text) 
        
        # 4. حفظ "شريط الأحداث" كصورة الدليل الرئيسية
        filename = f"timeline_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join("alerts_images", filename)
        cv2.imwrite(save_path, filmstrip) 

        save_alert(
            type=report.get('event'),
            description=json.dumps(report),
            confidence=99.0,
            image_filename=filename
        )
        print(f"💾 Report Saved: {report.get('event')} with Timeline.")

    except Exception as e:
        print(f"❌ Analysis Parsing Error: {e}")
    
    gemini_state["is_analyzing"] = False

# ==========================================
# 🎥 مولد الفيديو الرئيسي
# ==========================================
def generate_frames():
    global gemini_state, current_status
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    frame_count = 0
    current_trigger_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        frame_resized = cv2.resize(frame, RESIZE_DIM)

        # 1. كشف YOLO
        if frame_count % FRAME_SKIP_YOLO == 0:
            frame_for_processing = frame_resized.copy()
            processed_frame, _, objects, boxes = yolo_eye.detect_and_draw(frame_for_processing)
            
            is_danger = False
            danger_type = "Normal"
            
            for obj in objects:
                obj_lower = obj.lower()
                if any(word in obj_lower for word in ['violence', 'stabbing', 'weapon', 'gun', 'knife', 'criminal']):
                    is_danger = True
                    danger_type = obj.capitalize()

            with lock:
                current_status["boxes"] = boxes
                current_status["danger"] = is_danger
                current_status["danger_type"] = danger_type
            
            if is_danger:
                current_trigger_frame = processed_frame 

        # 2. الرسم والمعاينة للبث
        final_display_frame = frame_resized.copy()
        h, w, _ = final_display_frame.shape
        
        with lock:
            display_boxes = current_status["boxes"]
            display_danger = current_status["danger"]
            display_type = current_status["danger_type"]

        for (x1, y1, x2, y2, label, color) in display_boxes:
            cv2.rectangle(final_display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(final_display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if display_danger:
            cv2.rectangle(final_display_frame, (0,0), (w,h), (0,0,255), 4)
            cv2.putText(final_display_frame, f"THREAT: {display_type}", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if gemini_state["is_analyzing"]:
            cv2.putText(final_display_frame, "AI DEEP SCAN (30s Context)...", (w-220, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.circle(final_display_frame, (w-20, h-20), 6, (0, 255, 255), -1)

        # 3. إرسال التقرير السياقي
        current_time = time.time()
        if display_danger and \
           not gemini_state["is_analyzing"] and \
           (current_time - gemini_state["last_run"] > gemini_state["cooldown"]):
            
            gemini_state["is_analyzing"] = True
            gemini_state["last_run"] = current_time
            
            # أخذ لقطة ثابتة من الذاكرة المؤقتة (30 ثانية)
            buffer_snapshot = list(yolo_eye.frame_buffer)
            trigger_snapshot = current_trigger_frame if current_trigger_frame is not None else final_display_frame
            
            threading.Thread(
                target=gemini_worker, 
                args=(buffer_snapshot, trigger_snapshot, display_type)
            ).start()

        ret, buffer = cv2.imencode('.jpg', final_display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/history")
def read_history():
    return get_all_alerts()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)