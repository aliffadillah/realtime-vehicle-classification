from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2 as cv
from ultralytics import YOLO
import cpuinfo
from concurrent.futures import ThreadPoolExecutor
import os
import time
import numpy as np

# Pengaturan lingkungan
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# URL stream RTSP
rtsp_url = 'rtsp://admin:Pac1nk0!@158.140.176.70:9292/Streaming/Channels/101'

# Memuat model
model_path = 'detection/models/yolov8n_openvino_model/'
model = YOLO(model_path)

# Konfigurasi
conf_threshold = 0.1
target_class_ids = {0, 2, 3, 5, 7}  # ID untuk kendaraan dan orang
vehicle_names = {0: 'motor', 2: 'mobil', 3: 'bus', 5: 'truk', 7: 'orang'}
box_width, box_height = 1080, 720
line_y = 540  # Posisi garis di tengah

# Inisialisasi penghitung dan pelacak
vehicle_counts = {'motor': 0, 'mobil': 0, 'bus': 0, 'truk': 0}
person_count = 0
vehicle_trackers = {}

def get_bbox_info(box):
    x1, y1, x2, y2, score, class_id = box
    return int(x1), int(y1), int(x2), int(y2), int(class_id), score

def bbox_centroid(x1, x2, y1, y2):
    return int(x1 + ((x2 - x1) / 2)), int(y1 + ((y2 - y1) / 2))

def bbox_rectangle(x1, y1, x2, y2, frame):
    cv.rectangle(frame, (x1, y1), (x2, y2), (191, 64, 191), 3, cv.LINE_8)

def bbox_class_id_label(x1, y1, x2, y2, frame, class_id):
    vehicle_name = vehicle_names.get(class_id, 'unknown')
    cv.rectangle(frame, (x1, y1), (x1 + 60, y1 - 15), (255, 255, 255), -1, cv.LINE_8)
    cv.putText(frame, vehicle_name, (x1 + 5, y1 - 3), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)

def display_fps(frame, fps, cpu_type='', counts=None):
    font_scale = 0.6
    font = cv.FONT_HERSHEY_SIMPLEX
    fps_text = f'FPS: {round(fps, 2)}'
    height, width = frame.shape[:2]
    x_position, y_position = int(0.1 * width), int(0.1 * height)
    padding = 10

    # Tampilkan FPS
    cv.rectangle(frame, (x_position, y_position - 20), (x_position + 150, y_position + 30), (0, 0, 0), -1)
    cv.putText(frame, fps_text, (x_position, y_position), font, font_scale, (255, 255, 255), 2)

    # Tampilkan informasi CPU
    if cpu_type:
        y_position += 30
        cpu_text = f'CPU: {cpu_type}'
        cv.rectangle(frame, (x_position, y_position - 20), (x_position + 150, y_position + 30), (0, 0, 0), -1)
        cv.putText(frame, cpu_text, (x_position, y_position), font, font_scale, (255, 255, 255), 2)

    # Tampilkan jumlah
    if counts:
        y_position += 30
        for vehicle_type, count in counts.items():
            count_text = f'{vehicle_type.capitalize()}: {count}'
            cv.rectangle(frame, (x_position, y_position - 20), (x_position + 150, y_position + 30), (0, 0, 0), -1)
            cv.putText(frame, count_text, (x_position, y_position), font, font_scale, (255, 255, 255), 2)
            y_position += 30

def cpu_info():
    return cpuinfo.get_cpu_info()['brand_raw']

def update_vehicle_tracker(vehicle_id, class_id, bbox, velocity, frame):
    vehicle_name = vehicle_names.get(class_id, 'unknown')
    if vehicle_name not in vehicle_trackers:
        vehicle_trackers[vehicle_name] = {}
    if vehicle_id not in vehicle_trackers[vehicle_name]:
        vehicle_trackers[vehicle_name][vehicle_id] = {'counted': False, 'bbox': bbox, 'velocity': velocity}
    else:
        # Perbarui pelacak dengan bounding box dan kecepatan baru
        vehicle_trackers[vehicle_name][vehicle_id]['bbox'] = bbox
        vehicle_trackers[vehicle_name][vehicle_id]['velocity'] = velocity

def generate_vehicle_id(bbox, velocity):
    # Gabungkan bbox dan kecepatan untuk menghasilkan ID unik
    return hash(tuple(bbox) + tuple(velocity))

def calculate_velocity(prev_bbox, curr_bbox):
    # Hitung kecepatan berdasarkan pergerakan pusat bounding box
    prev_centroid = bbox_centroid(*prev_bbox[:2], *prev_bbox[2:])
    curr_centroid = bbox_centroid(*curr_bbox[:2], *curr_bbox[2:])
    velocity = (curr_centroid[0] - prev_centroid[0], curr_centroid[1] - prev_centroid[1])
    return velocity

def process_frame(frame, results):
    height, width = frame.shape[:2]
    x_center = width // 2
    y_center = height // 2
    x1 = x_center - box_width // 2
    y1 = y_center - box_height // 2
    x2 = x_center + box_width // 2
    y2 = y_center + box_height // 2

    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.line(frame, (x1, y1 + line_y), (x2, y1 + line_y), (0, 0, 255), 2)

    global vehicle_counts, person_count, vehicle_trackers

    for result in results:
        for box in result.boxes.data.tolist():
            x1_box, y1_box, x2_box, y2_box, class_id, score = get_bbox_info(box)
            vehicle_name = vehicle_names.get(class_id, 'unknown')

            if class_id in target_class_ids and score >= conf_threshold:
                if x1 <= x1_box <= x2 and y1 <= y1_box <= y2:
                    bbox = (x1_box, y1_box, x2_box, y2_box)
                    bbox_rectangle(x1_box, y1_box, x2_box, y2_box, frame)
                    bbox_class_id_label(x1_box, y1_box, x2_box, y2_box, frame, class_id)

                    # Menghasilkan ID kendaraan unik berdasarkan bbox dan kecepatan
                    prev_bbox = vehicle_trackers.get(vehicle_name, {}).get('bbox', bbox)
                    velocity = calculate_velocity(prev_bbox, bbox)
                    vehicle_id = generate_vehicle_id(bbox, velocity)

                    # Perbarui pelacak
                    update_vehicle_tracker(vehicle_id, class_id, bbox, velocity, frame)

                    if vehicle_name == 'orang':
                        if not vehicle_trackers[vehicle_name][vehicle_id]['counted']:
                            person_count += 1
                            vehicle_trackers[vehicle_name][vehicle_id]['counted'] = True
                    else:
                        if not vehicle_trackers[vehicle_name][vehicle_id]['counted']:
                            if y1_box < y1 + line_y and y2_box > y1 + line_y:
                                vehicle_counts[vehicle_name] += 1
                                vehicle_trackers[vehicle_name][vehicle_id]['counted'] = True
                                print(f"{vehicle_name.capitalize()} melewati garis Y")

    return frame

def gen_frames():
    cap = cv.VideoCapture(rtsp_url)
    
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    cpu_type = cpu_info()

    with ThreadPoolExecutor(max_workers=2) as executor:
        frame_skip_interval = 2
        frame_counter = 0
        retry_count = 0
        max_retries = 5

        while retry_count < max_retries:
            if not cap.isOpened():
                print("Menyambung kembali ke stream RTSP...")
                cap = cv.VideoCapture(rtsp_url)
                retry_count += 1
                time.sleep(2)
                continue

            retry_count = 0
            ret, frame = cap.read()

            if not ret:
                print("Frame tidak diterima, mencoba kembali...")
                retry_count += 1
                time.sleep(2)
                continue

            frame_counter += 1
            if frame_counter % frame_skip_interval != 0:
                continue

            start_time = time.time()

            results = model(frame)
            frame = process_frame(frame, results)

            end_time = time.time()
            fps = 1 / (end_time - start_time)

            # Tampilkan tipe CPU dan FPS
            display_fps(frame, fps, cpu_type, vehicle_counts)

            _, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

def index(request):
    return render(request, 'detection/index.html')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
