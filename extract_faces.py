# extract_faces_fixed.py
import os
import cv2
import pandas as pd
import csv
import numpy as np

# --------------------
# CONFIG
# --------------------
ROOT_DIR = r"C:\Users\rudra\Desktop\My_Pain_Study"
ALL_FACES_DIR = os.path.join(ROOT_DIR, "all_faces_dataset")
CSV_PATH = os.path.join(ALL_FACES_DIR, "face_metadata.csv")
IMG_SIZE = 224
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

os.makedirs(ALL_FACES_DIR, exist_ok=True)
os.makedirs(os.path.join(ALL_FACES_DIR, "normal"), exist_ok=True)
os.makedirs(os.path.join(ALL_FACES_DIR, "pain"), exist_ok=True)

# initialize csv if not exists
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["subj_id", "video_file", "frame_idx", "timestamp", "label", "image_path"])

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def process_subject(subject_dir):
    print(f"\nProcessing subject folder: {subject_dir}")

    sync_csv_path = os.path.join(subject_dir, "synchronized.csv")
    video_files = [f for f in os.listdir(subject_dir) if f.lower().endswith(".mp4")]
    ts_files = [f for f in os.listdir(subject_dir) if f.lower().endswith("_timestamps.csv")]

    if not os.path.exists(sync_csv_path):
        print("  synchronized.csv not found.")
        return
    if not video_files or not ts_files:
        print("  Video or timestamps not found.")
        return

    sync_df = pd.read_csv(sync_csv_path)
    if 'ts' not in sync_df.columns:
        print("  synchronized.csv missing 'ts' column.")
        return
    # ensure last_marker present
    if 'last_marker' not in sync_df.columns:
        sync_df['last_marker'] = sync_df.get('sample', "")  # fallback

    subj_id = os.path.basename(subject_dir)

    for video_file, ts_file in zip(sorted(video_files), sorted(ts_files)):
        video_path = os.path.join(subject_dir, video_file)
        ts_path = os.path.join(subject_dir, ts_file)
        ts_df = pd.read_csv(ts_path)
        # timestamp column could be named differently
        vcol = ts_df.columns[0]
        v_ts = ts_df[vcol].astype(float).values

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # compute frame timestamp robustly:
            if frame_idx < len(v_ts):
                frame_ts = float(v_ts[frame_idx])
            else:
                # fallback estimate by FPS
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_ts = v_ts[-1] + (frame_idx - len(v_ts) + 1) * (1.0/fps)

            # find last marker at or before frame_ts
            pos = np.searchsorted(sync_df['ts'].values, frame_ts, side='right')
            if pos == 0:
                marker_text = str(sync_df['last_marker'].iloc[0]) if len(sync_df)>0 else ""
            else:
                marker_text = str(sync_df['last_marker'].iloc[pos-1])

            marker_text_l = marker_text.lower()
            if 'pain' in marker_text_l:
                label = 'pain'
            elif 'normal' in marker_text_l:
                label = 'normal'
            else:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
            if len(faces) == 0:
                frame_idx += 1
                continue
            # choose largest face
            faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            x,y,w,h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

            out_name = f"{subj_id}_{os.path.splitext(video_file)[0]}_frame{frame_idx:05d}.png"
            out_path = os.path.join(ALL_FACES_DIR, label, out_name)
            cv2.imwrite(out_path, face_img)

            with open(CSV_PATH, "a", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([subj_id, video_file, frame_idx, frame_ts, label, out_path])

            saved += 1
            # optionally skip a few frames to reduce redundancy:
            # frame_idx += frame_skip (but here we increment by one for full extraction)
            frame_idx += 1

        cap.release()
        print(f"  {video_file}: saved {saved} faces")

    print(f"✅ Finished subject: {subject_dir}")

def main():
    for root, dirs, files in os.walk(ROOT_DIR):
        for d in dirs:
            subj_dir = os.path.join(root, d)
            if os.path.exists(os.path.join(subj_dir, "synchronized.csv")):
                process_subject(subj_dir)
    print("\n✅ All faces + metadata saved into:", ALL_FACES_DIR)
    print("Metadata CSV:", CSV_PATH)

if __name__ == "__main__":
    main()
