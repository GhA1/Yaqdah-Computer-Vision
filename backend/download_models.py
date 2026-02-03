import os
import requests

# المجلد المستهدف
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# روابط الملفات (TSM ResNet50 - Kinetics400)
files = {
    "tsm_config.py": "https://raw.githubusercontent.com/open-mmlab/mmaction/main/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400_rgb.py",
    "tsm_checkpoint.pth": "https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400_rgb_20220831-64d69186.pth"
}

def download_file(url, filename):
    local_path = os.path.join(SAVE_DIR, filename)
    print(f"⬇️ Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Saved to {local_path}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

if __name__ == "__main__":
    print("🚀 Starting Model Download...")
    for name, url in files.items():
        download_file(url, name)
    print("🎉 Done! Files are ready in the 'checkpoints' folder.")