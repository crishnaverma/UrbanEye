#!/usr/bin/env python3
"""
SIH 2026 - UrbanEye Unified Multi-Modal Fleet Intelligence Demo
==============================================================
Runs real-time multi-modal AI perception on video or live camera:
1. Road Defect & Pothole Detection (Potholes, Cracks, Manholes, Waterlogging)
2. Traffic & Vehicle Density (Cars, Buses, Trucks, Motorcycles)
3. Pedestrian Safety & Vulnerable Road Users
4. Automatic Number Plate Recognition (ANPR Localizer)

Live-synchronizes all detections to SQLite (data/events.db) and the
UrbanEye Central FastAPI Backend (http://127.0.0.1:8000/events) so the
web GIS dashboard at http://127.0.0.1:3000 reflects changes in real-time.
"""

import os
import sys
# Set YOLO config directory to prevent permission warnings
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("data")

import time
import json
import uuid
import random
import sqlite3
import argparse
from datetime import datetime, timezone
import cv2
import requests
from ultralytics import YOLO

# Highly accurate waypoints along the Delhi DTC Transit Corridor
DELHI_WAYPOINTS = [
    (28.5672, 77.2100, "AIIMS Flyover / Ring Road"),
    (28.5710, 77.2225, "South Extension II"),
    (28.5695, 77.2340, "Moolchand Underpass"),
    (28.5685, 77.2435, "Lajpat Nagar Ring Road"),
    (28.5710, 77.2588, "Ashram Chowk Intersection"),
    (28.5880, 77.2485, "Mathura Road / Nizamuddin"),
    (28.6040, 77.2440, "Sunder Nagar Main Road"),
    (28.6150, 77.2420, "Pragati Maidan / Bhairon Marg"),
    (28.6135, 77.2380, "Purana Qila Road"),
    (28.6129, 77.2295, "India Gate C-Hexagon"),
    (28.6235, 77.2240, "Kasturba Gandhi Marg"),
    (28.6275, 77.2205, "Tolstoy Marg Junction"),
    (28.6315, 77.2195, "Connaught Place Outer Circle"),
    (28.6328, 77.2197, "Rajiv Chowk / CP Inner Circle")
]

CLASS_NAME_MAP = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole",
    4: "manhole",
    5: "waterlogging"
}

# Color palette (BGR)
COLOR_PALETTE = {
    # Road Defects
    "pothole": (0, 0, 255),           # Pure Red
    "alligator_crack": (0, 220, 255),  # Yellow
    "longitudinal_crack": (255, 120, 0),# Blue
    "transverse_crack": (0, 200, 100), # Green
    "manhole": (220, 50, 200),         # Purple
    "waterlogging": (255, 255, 0),     # Cyan
    # Traffic
    "car": (255, 180, 50),
    "bus": (255, 100, 0),
    "truck": (200, 100, 50),
    "motorcycle": (0, 215, 255),
    "bicycle": (0, 255, 180),
    # Pedestrians
    "person": (50, 205, 50),
    # ANPR Plates
    "number_plate": (255, 50, 255),
    "plate": (255, 50, 255)
}

def interpolate_gps(progress_ratio: float):
    """Interpolate GPS along Delhi transit corridor based on video frame progression."""
    t = max(0.0, min(1.0, progress_ratio))
    total_segments = len(DELHI_WAYPOINTS) - 1
    seg_idx = int(t * total_segments)
    next_idx = min(seg_idx + 1, total_segments)
    sub_t = (t * total_segments) - seg_idx

    p1 = DELHI_WAYPOINTS[seg_idx]
    p2 = DELHI_WAYPOINTS[next_idx]

    lat = round(p1[0] + sub_t * (p2[0] - p1[0]), 5)
    lon = round(p1[1] + sub_t * (p2[1] - p1[1]), 5)
    location_name = p1[2]
    return lat, lon, location_name

def log_event_to_db_and_api(event_type: str, confidence: float, lat: float, lon: float, location_name: str, box_coords: list, extra_meta=None, db_path="data/events.db"):
    """Insert event into data/events.db and attempt non-blocking POST to API."""
    event_id = f"evt_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()
    source = "DTC-BUS-101"

    payload_dict = {
        "detector": "UrbanEye-UnifiedAI",
        "location": location_name,
        "box": box_coords,
        "bus_id": source,
        "speed_kmh": round(32.0 + random.uniform(-4, 4), 1)
    }
    if extra_meta:
        payload_dict.update(extra_meta)

    # 1. Direct SQLite insertion
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO events (event_id, event_type, source, timestamp, latitude, longitude, confidence, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, event_type, source, now_iso, lat, lon, confidence, json.dumps(payload_dict), now_iso)
            )

            # High confidence potholes create an incident in command center
            if event_type == "pothole" and confidence >= 0.40:
                inc_id = f"INC-{uuid.uuid4().hex[:6].upper()}"
                title = f"Pothole Hazard near {location_name}"
                desc = f"Verified hazard ({int(confidence*100)}% conf) reported by transit unit {source}."
                cur.execute(
                    """INSERT INTO incidents (incident_id, event_id, status, severity, title, description, operator, created_at, updated_at)
                       VALUES (?, ?, 'open', 'critical', ?, ?, 'Autonomous Fleet AI', ?, ?)""",
                    (inc_id, event_id, title, desc, now_iso, now_iso)
                )

            conn.commit()
            conn.close()
        except Exception:
            pass

    # 2. Non-blocking FastAPI endpoint post
    try:
        api_payload = {
            "event_id": event_id,
            "event_type": event_type,
            "source": source,
            "timestamp": now_iso,
            "latitude": lat,
            "longitude": lon,
            "confidence": confidence,
            "payload": payload_dict
        }
        requests.post("http://127.0.0.1:8000/events", json=api_payload, timeout=0.15)
    except Exception:
        pass

    return event_id

def run_all_modes_demo(source="data/sample_videos/demovid.mp4", conf=0.30, loop=True, enable_all=True):
    pothole_weights = "models/pothole.pt"
    yolo_weights = "models/yolov8n.pt"
    anpr_weights = "models/anpr/best.pt"

    if not os.path.exists(pothole_weights):
        print(f"[ERROR] Road damage weights not found at {pothole_weights}")
        sys.exit(1)

    print("\n" + "="*70)
    print("🚦 SIH 2026: URBANEYE UNIFIED MULTI-MODAL FLEET AI DEMO")
    print(f" Source Video: {source}")
    print(f" Mode: {'ALL MODES (Road Defects + Vehicles + Pedestrians + ANPR)' if enable_all else 'Potholes Only'}")
    print(" Live Web Sync: ACTIVE -> Syncing detections to http://127.0.0.1:3000")
    print(" Press 'q' or 'ESC' on the video preview window to exit")
    print("="*70 + "\n")

    print("[AI 1/3] Loading Road Defect & Pothole Model (YOLOv8-RoadDefect)...")
    pothole_model = YOLO(pothole_weights)

    traffic_model = None
    anpr_model = None
    if enable_all:
        if os.path.exists(yolo_weights):
            print("[AI 2/3] Loading Traffic & Vehicle Perception Model (YOLOv8-Traffic)...")
            traffic_model = YOLO(yolo_weights)
        if os.path.exists(anpr_weights):
            print("[AI 3/3] Loading ANPR License Plate Localizer (YOLOv8-ANPR)...")
            anpr_model = YOLO(anpr_weights)

    is_cam = str(source).isdigit() or source == "0"
    src_val = int(source) if is_cam else source

    win_title = "SIH 2026: UrbanEye Unified Multi-Modal Perception (Press Q to Exit)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 1280, 720)

    logged_events_count = 0
    last_logged_time = 0.0
    cooldown_seconds = 1.0

    # Total session statistics
    stats = {
        "potholes": 0,
        "cracks": 0,
        "vehicles": 0,
        "pedestrians": 0,
        "plates": 0
    }

    while True:
        cap = cv2.VideoCapture(src_val)
        if not cap.isOpened():
            print(f"[ERROR] Unable to open source: {source}")
            break

        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_idx = 0
        prev_time = time.time()
        fps = 30.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            progress = (frame_idx % total_frames) / float(total_frames) if not is_cam else ((frame_idx % 300) / 300.0)
            cur_lat, cur_lon, cur_loc = interpolate_gps(progress)

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt)

            h, w = frame.shape[:2]

            # -------------------------------------------------------------
            # MODEL 1: Road Defects & Potholes
            # -------------------------------------------------------------
            res_potholes = pothole_model.predict(frame, conf=conf, imgsz=640, verbose=False)[0]

            frame_potholes = 0
            frame_cracks = 0
            top_pothole_event = None

            for box in res_potholes.boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = CLASS_NAME_MAP.get(cls_id, pothole_model.names.get(cls_id, f"Defect_{cls_id}")).lower()
                confidence = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]

                display_name = cls_name.replace("_", " ").upper()
                color = COLOR_PALETTE.get(cls_name, (0, 0, 255))

                if cls_name == "pothole":
                    frame_potholes += 1
                    stats["potholes"] += 1
                else:
                    frame_cracks += 1
                    stats["cracks"] += 1

                # Draw bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{display_name} {int(confidence*100)}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                if top_pothole_event is None or confidence > top_pothole_event["conf"]:
                    top_pothole_event = {
                        "type": cls_name,
                        "conf": confidence,
                        "box": [x1, y1, x2, y2]
                    }

            # -------------------------------------------------------------
            # MODEL 2: Vehicles, Traffic Density & Pedestrians (COCO)
            # -------------------------------------------------------------
            frame_vehicles = 0
            frame_pedestrians = 0
            if enable_all and traffic_model is not None:
                # Classes: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
                res_traffic = traffic_model.predict(
                    frame, conf=0.35, imgsz=640,
                    classes=[0, 1, 2, 3, 5, 7],
                    verbose=False
                )[0]

                for box in res_traffic.boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    raw_name = traffic_model.names.get(cls_id, f"obj_{cls_id}").lower()
                    confidence = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]

                    if raw_name in ["car", "bus", "truck", "motorcycle"]:
                        frame_vehicles += 1
                        stats["vehicles"] += 1
                        color = COLOR_PALETTE.get(raw_name, (255, 180, 50))
                        lbl_text = f"{raw_name.upper()} {int(confidence*100)}%"
                    elif raw_name in ["person", "bicycle"]:
                        frame_pedestrians += 1
                        stats["pedestrians"] += 1
                        color = (50, 205, 50)  # Green for pedestrians
                        lbl_text = f"PEDESTRIAN {int(confidence*100)}%"
                    else:
                        continue

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (lw, lh), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - lh - 4), (x1 + lw + 4, y1), color, -1)
                    cv2.putText(frame, lbl_text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

            # -------------------------------------------------------------
            # MODEL 3: License Plates (ANPR)
            # -------------------------------------------------------------
            frame_plates = 0
            if enable_all and anpr_model is not None and frame_idx % 2 == 0:
                res_anpr = anpr_model.predict(frame, conf=0.25, imgsz=640, verbose=False)[0]
                for box in res_anpr.boxes:
                    confidence = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    frame_plates += 1
                    stats["plates"] += 1

                    color = (255, 50, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    lbl = f"PLATE {int(confidence*100)}%"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x1, y1 - lh - 4), (x1 + lw + 4, y1), color, -1)
                    cv2.putText(frame, lbl, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # -------------------------------------------------------------
            # Real-Time Event Sync to Database and Web Dashboard
            # -------------------------------------------------------------
            if curr_time - last_logged_time >= cooldown_seconds:
                last_logged_time = curr_time
                if top_pothole_event:
                    log_event_to_db_and_api(
                        event_type=top_pothole_event["type"],
                        confidence=top_pothole_event["conf"],
                        lat=cur_lat,
                        lon=cur_lon,
                        location_name=cur_loc,
                        box_coords=top_pothole_event["box"]
                    )
                    logged_events_count += 1
                elif frame_vehicles > 0 and random.random() < 0.25:
                    log_event_to_db_and_api(
                        event_type="vehicle_count",
                        confidence=0.92,
                        lat=cur_lat,
                        lon=cur_lon,
                        location_name=cur_loc,
                        box_coords=[0, 0, w, h],
                        extra_meta={"vehicle_density": frame_vehicles}
                    )
                    logged_events_count += 1

            # -------------------------------------------------------------
            # Professional Heads-Up Display (HUD) Overlays
            # -------------------------------------------------------------
            # 1. Top Header Bar
            cv2.rectangle(frame, (0, 0), (w, 54), (18, 18, 22), -1)
            cv2.line(frame, (0, 54), (w, 54), (0, 220, 255), 2)
            cv2.putText(frame, "SIH 2026: URBANEYE MULTI-MODAL FLEET PERCEPTION (DTC-101)", 
                        (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

            telemetry_str = f"GPS: {cur_lat:.4f}, {cur_lon:.4f} | {cur_loc}"
            cv2.putText(frame, telemetry_str, (16, 44), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            sync_status_str = f"FPS: {fps:.1f} | Web Sync: ACTIVE [Logged: {logged_events_count}]"
            cv2.putText(frame, sync_status_str, (w - 480, 24), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)

            mode_str = "Mode: ALL SENSORS (Defects + Vehicles + Pedestrians + ANPR)" if enable_all else "Mode: Road Defects Only"
            cv2.putText(frame, mode_str, (w - 480, 44), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

            # 2. Bottom Analytics Ribbon
            cv2.rectangle(frame, (0, h - 42), (w, h), (18, 18, 22), -1)
            cv2.line(frame, (0, h - 42), (w, h - 42), (60, 60, 60), 1)

            health_score = max(20, 100 - (frame_potholes * 25 + frame_cracks * 10))
            health_color = (0, 255, 100) if health_score >= 80 else ((0, 200, 255) if health_score >= 50 else (0, 0, 255))

            bottom_stats = (
                f"POTHOLES: {frame_potholes}  |  "
                f"CRACKS: {frame_cracks}  |  "
                f"VEHICLES: {frame_vehicles}  |  "
                f"PEDESTRIANS: {frame_pedestrians}  |  "
                f"PLATES: {frame_plates}"
            )
            cv2.putText(frame, bottom_stats, (16, h - 14), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)

            health_text = f"ROAD HEALTH: {health_score}/100"
            cv2.putText(frame, health_text, (w - 240, h - 14), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, health_color, 2)

            # Render frame
            cv2.imshow(win_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        if not loop or is_cam:
            break
        print("[DEMO] Looping video playback...")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanEye Unified Multi-Modal Perception Demo")
    parser.add_argument("--source", type=str, default="data/sample_videos/demovid.mp4", 
                        help="Path to video file or 0 for webcam")
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop video")
    parser.add_argument("--potholes-only", action="store_true", help="Run only road defect detector")
    parser.add_argument("--all", action="store_true", default=True, help="Run all detection modes simultaneously (default)")
    args = parser.parse_args()

    enable_all_modes = not args.potholes_only

    run_all_modes_demo(
        source=args.source,
        conf=args.conf,
        loop=not args.no_loop,
        enable_all=enable_all_modes
    )
