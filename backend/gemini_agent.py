import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import cv2
import warnings
import json
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# تحميل المفتاح
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY is missing!")
else:
    genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel('gemini-2.5-flash')
except:
    model = genai.GenerativeModel('gemini-pro-vision')

def generate_security_report(frames_list, detected_type):
    """
    توليد تقرير أمني "ناقد" يتحقق من صحة التنبيه
    """
    print("📝 Gemini Investigating Scene...")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    pil_images = []
    for frame in frames_list:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb_frame))

    # =========================================================
    # 🕵️‍♂️ البرومبت الناقد (The Skeptical Investigator)
    # =========================================================
    prompt = f"""
    Role: Senior Security Analyst Verification AI.
    
    --- INPUT ---
    • Sensor Trigger: "{detected_type}"
    • Timestamp: {current_time}
    -------------

    TASK:
    Verify if this trigger is REAL or a FALSE ALARM. 
    Computer vision sensors make mistakes. You are the second opinion.

    CRITICAL RULES:
    1. **BE HONEST:** If you see an empty street, people just walking, or shadows -> SAY IT IS A FALSE ALARM.
    2. **DO NOT TRUST THE SENSOR:** Just because it says "{detected_type}" does NOT mean it's true.
    3. **CLASSIFICATION:**
       - If visual evidence of crime exists -> Name the crime (e.g. Armed Robbery, Assault).
       - If ambiguous/unclear -> "Suspicious Activity".
       - If nothing is happening -> "False Alarm / Normal Routine".

    Output Format (Strict JSON):
    {{
        "event": "Your Verdict (e.g. False Alarm, Assault, Loitering)",
        "risk_level": "Low/Medium/High", 
        "person_count": "Count",
        "suspect_desc": "Description OR 'None'",
        "details": "Direct observation. E.g., 'Sensor triggered violence, but visual review shows only pedestrians walking calmly. No threat visible.'",
        "actions": ["Action 1", "Action 2"]
    }}
    """

    try:
        response = model.generate_content(pil_images + [prompt])
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        # =========================================================
        # 🚨 تعديل كود الطوارئ ليكون صادقاً أيضاً
        # =========================================================
        if "429" in error_msg:
            print("⏳ Quota Limit. Saving unverified report.")
            return json.dumps({
                "event": f"Unverified Alert ({detected_type})", # نوضح أنه غير مؤكد
                "risk_level": "Medium", # نزلناه من High إلى Medium
                "person_count": "Unknown",
                "suspect_desc": "System Busy",
                "details": f"Sensor detected {detected_type} but AI could not verify due to high traffic. Visual check recommended.",
                "actions": ["Manually Check Camera", "Monitor Sector"]
            })
        else:
            print(f"❌ Gemini Error: {error_msg}")
            return json.dumps({
                "event": "System Error",
                "risk_level": "Unknown",
                "person_count": "0",
                "suspect_desc": "N/A",
                "details": "Connection failed.",
                "actions": ["Check Network"]
            })