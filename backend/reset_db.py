import sqlite3
import os

DB_NAME = "smart_guard.db"

def reset_database():
    if os.path.exists(DB_NAME):
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            # حذف الجدول بالكامل
            cursor.execute("DROP TABLE IF EXISTS alerts")
            conn.commit()
            print("🗑️  تم حذف جميع التقارير القديمة بنجاح!")
            
            # (اختياري) حذف الصور أيضاً لتوفير المساحة
            folder = "alerts_images"
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if file.endswith('.jpg') or file.endswith('.png'):
                        os.remove(file_path)
                print("🖼️  تم حذف صور الأدلة القديمة.")

        except Exception as e:
            print(f"❌ حدث خطأ: {e}")
        finally:
            conn.close()
    else:
        print("⚠️ قاعدة البيانات غير موجودة أصلاً.")

if __name__ == "__main__":
    reset_database()
    print("✅ جاهز! الآن شغل main.py وسيتم إنشاء قاعدة بيانات جديدة ونظيفة.")