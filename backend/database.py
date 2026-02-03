import sqlite3

DB_NAME = "smart_guard.db"

def init_db():
    # check_same_thread=False ضروري لتعدد العمليات
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    
    # ✅ تفعيل وضع WAL لمنع قفل القاعدة أثناء الكتابة
    cursor.execute("PRAGMA journal_mode=WAL;")
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            description TEXT,
            confidence REAL,
            image_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_alert(type, description, confidence, image_filename):
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO alerts (type, description, confidence, image_path) VALUES (?, ?, ?, ?)",
                       (type, description, confidence, image_filename))
        conn.commit()
    except Exception as e:
        print(f"❌ DB Error: {e}")
    finally:
        conn.close()

def get_all_alerts():
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"❌ DB Read Error: {e}")
        return []
    finally:
        conn.close()

init_db()