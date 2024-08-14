from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2 as cv
from ultralytics import YOLO
import cpuinfo
from concurrent.futures import ThreadPoolExecutor
import os
import time  # Tambahkan ini untuk menghitung waktu

# Pengaturan lingkungan untuk menghindari konflik dengan pustaka
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# RTSP stream URL
rtsp_url = 'rtsp://admin:Pac1nk0!@158.140.176.70:9292/Streaming/Channels/101'

# Load model
model_path = 'detection/models/yolov8n_openvino_model/'
model = YOLO(model_path)

# Konfigurasi model dan deteksi
conf_threshold = 0.1
target_class_ids = {0, 2, 3, 5, 7}  # IDs untuk kendaraan
box_width, box_height = 1080, 720
line_y = 360  # Posisi garis di tengah

def get_bbox_info(box):
    x1, y1, x2, y2, score, class_id = box
    return int(x1), int(y1), int(x2), int(y2), int(class_id), score

def bbox_centroid(x1, x2, y1, y2, frame):
    x_point = int(x1 + ((x2 - x1) / 2))
    y_point = int(y1 + ((y2 - y1) / 2))
    cv.circle(frame, (x_point, y_point), 2, (255, 255, 255), 2, cv.LINE_8)
    return x_point, y_point

def bbox_rectangle(x1, y1, x2, y2, frame):
    cv.rectangle(frame, (x1, y1), (x2, y2), (191, 64, 191), 3, cv.LINE_8)

def bbox_class_id_label(x1, y1, x2, y2, frame, class_id):
    cv.rectangle(frame, (x1, y1), (x1 + 30, y1 - 15), (255, 255, 255), -1, cv.LINE_8)
    cv.putText(frame, f'ID:{class_id}', (x1 + 5, y1 - 3), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)

def display_fps(frame, fps, cpu_type=''):
    font_scale = 0.6
    font = cv.FONT_HERSHEY_SIMPLEX
    fps_text = f'FPS: {round(fps, 2)}'
    (text_width, text_height) = cv.getTextSize(fps_text, font, fontScale=font_scale, thickness=1)[0]
    height, width = frame.shape[:2]
    x_position, y_position = int(0.1 * width), int(0.1 * height)
    padding = 10

    rect_coord = ((x_position, y_position), (x_position + text_width + padding, y_position - text_height - padding))
    cv.rectangle(frame, rect_coord[0], rect_coord[1], (0, 0, 0), -1, cv.LINE_8)
    text_coord = (x_position + int(padding / 2), y_position - int(padding / 2))
    cv.putText(frame, fps_text, text_coord, font, fontScale=font_scale, color=(255, 255, 255), thickness=2)

    if cpu_type:
        cpu_text = f'CPU: {cpu_type}'
        (text_width, text_height) = cv.getTextSize(cpu_text, font, fontScale=font_scale, thickness=1)[0]
        y_position += text_height + padding
        rect_coord = ((x_position, y_position), (x_position + text_width + padding, y_position - text_height - padding))
        cv.rectangle(frame, rect_coord[0], rect_coord[1], (0, 0, 0), -1, cv.LINE_8)
        text_coord = (x_position + int(padding / 2), y_position - int(padding / 2))
        cv.putText(frame, cpu_text, text_coord, font, fontScale=font_scale, color=(255, 255, 255), thickness=2)

def cpu_info():
    return cpuinfo.get_cpu_info()['brand_raw']

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

    for result in results:
        for box in result.boxes.data.tolist():
            x1_box, y1_box, x2_box, y2_box, class_id, score = get_bbox_info(box)
            if class_id in target_class_ids and score >= conf_threshold:
                if x1 <= x1_box <= x2 and y1 <= y1_box <= y2:
                    bbox_rectangle(x1_box, y1_box, x2_box, y2_box, frame)
                    bbox_class_id_label(x1_box, y1_box, x2_box, y2_box, frame, class_id)
                    x_centroid, y_centroid = bbox_centroid(x1_box, x2_box, y1_box, y2_box, frame)
                    if y_centroid >= y1 + line_y:
                        print("Kendaraan melewati garis Y")
    return frame

def gen_frames():
    cap = cv.VideoCapture(rtsp_url)
    
    # Set resolusi input untuk mengurangi beban pemrosesan
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    cpu_type = cpu_info()

    # Konfigurasi untuk ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        frame_skip_interval = 2  # Hanya memproses setiap frame ke-2
        frame_counter = 0
        retry_count = 0
        max_retries = 5  # Maksimal 5 kali percobaan ulang jika terjadi kesalahan

        while retry_count < max_retries:  # Coba ulang jika ada kegagalan
            if not cap.isOpened():
                print("Reconnecting to RTSP stream...")
                cap = cv.VideoCapture(rtsp_url)
                retry_count += 1
                time.sleep(2)  # Beri waktu sebelum mencoba kembali
                continue

            start_time = time.time()  # Waktu mulai pemrosesan frame

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame, reconnecting...")
                retry_count += 1
                cap.release()
                cap = cv.VideoCapture(rtsp_url)
                time.sleep(2)  # Beri waktu sebelum mencoba kembali
                continue

            frame_counter += 1
            if frame_counter % frame_skip_interval != 0:
                continue  # Lewati frame untuk mengurangi beban pemrosesan

            try:
                # Lakukan inferensi secara asinkron
                future = executor.submit(model, frame, stream=True, device="cpu")
                results = future.result()

                # Proses frame
                frame = process_frame(frame, results)

                # Hitung FPS berdasarkan waktu yang berlalu
                elapsed_time = time.time() - start_time
                fps = 1 / elapsed_time

                # Pastikan FPS tidak melebihi 25
                if fps > 25:
                    time.sleep(1 / 25 - elapsed_time)

                # Tampilkan FPS pada frame
                display_fps(frame, fps, cpu_type)

                # Encode frame ke format JPEG
                ret, buffer = cv.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                # Reset retry counter setelah sukses memproses frame
                retry_count = 0

            except Exception as e:
                print(f"Error occurred during processing: {e}")
                retry_count += 1
                cap.release()
                cap = cv.VideoCapture(rtsp_url)
                time.sleep(2)  # Beri waktu sebelum mencoba kembali
                continue

    cap.release()


def index(request):
    return render(request, 'detection/index.html')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
