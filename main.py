import os
import io
import cv2
import jwt
import csv
import uuid
import time
import json
import math
import base64
import queue
import shutil
import struct
import hashlib
import logging
import asyncio
import tempfile
import datetime
import threading
import traceback
import numpy as np
import urllib.request
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps

import redis.asyncio as aioredis
import boto3
from botocore.client import Config
import psycopg2
import psycopg2.extras
import psycopg2.pool
from psycopg2.pool import ThreadedConnectionPool

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException,
    Depends, BackgroundTasks, UploadFile, File, Header,
    Request, Query, Path as FPath, Form, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocketState

import torch
import torchvision
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
from deepface import DeepFace
from paddleocr import PaddleOCR
import pyaudio
import wave
import librosa
import librosa.display
import soundfile as sf
from scipy.spatial.distance import cosine, euclidean
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import skimage
from skimage import exposure, filters, morphology

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("argus")

DB_URL = os.environ["SUPABASE_DB_URL"]
UPSTASH_REDIS_URL = os.environ["UPSTASH_REDIS_URL"]
UPSTASH_REDIS_TOKEN = os.environ["UPSTASH_REDIS_TOKEN"]
R2_ACCOUNT_ID = os.environ["CF_R2_ACCOUNT_ID"]
R2_ACCESS_KEY = os.environ["CF_R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["CF_R2_SECRET_KEY"]
R2_BUCKET = os.environ["CF_R2_BUCKET"]
JWT_SECRET = os.environ["JWT_SECRET"]
MASTER_API_KEY = os.environ["MASTER_API_KEY"]
PORT = int(os.environ.get("PORT", 8000))
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 8))
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", 2))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", 10))
SNAPSHOT_QUALITY = int(os.environ.get("SNAPSHOT_QUALITY", 85))
STREAM_QUALITY = int(os.environ.get("STREAM_QUALITY", 75))

FACE_SIMILARITY_THRESHOLD = 0.42
FACE_DETECTION_CONFIDENCE = 0.75
WEAPON_CONFIDENCE_THRESHOLD = 0.58
VEHICLE_CONFIDENCE_THRESHOLD = 0.50
PERSON_CONFIDENCE_THRESHOLD = 0.48
PLATE_OCR_CONFIDENCE = 0.60
EMOTION_INTERVAL_FRAMES = 8
PLATE_SCAN_INTERVAL_FRAMES = 4
POSE_ANALYSIS_INTERVAL_FRAMES = 3
AUDIO_THREAT_INTERVAL_FRAMES = 15
MIN_TRACK_FRAMES = 8
HISTORY_LEN = 35
MAX_PLAUSIBLE_KPH = 260
MIN_PLAUSIBLE_KPH = 2
LOITER_TIME_SECONDS = 35
CROWD_DENSITY_ALERT_THRESHOLD = 12
RUNNING_SPEED_THRESHOLD = 4.2
FIGHT_MOTION_THRESHOLD = 85.0
ATTENTION_HEATMAP_DECAY = 0.96
GAIT_SIGNATURE_LEN = 32
NIGHT_MODE_THRESHOLD = 45
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHUNK_SIZE = 2048
AUDIO_OVERLAP = 512
MAX_AUDIO_BUFFER = 64
PLATE_MIN_CHARS = 3
PLATE_MAX_CHARS = 12
HEATMAP_GRID_SIZE = 20
BEHAVIOR_WINDOW_SECONDS = 60
MAX_SPEED_HISTORY = 50
GAIT_WINDOW = 20
DEMOGRAPHIC_AGE_BINS = [(0, 12), (13, 17), (18, 25), (26, 35), (36, 50), (51, 65), (66, 100)]

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle", 8: "boat"}
PERSON_CLASS = 0
WEAPON_KEYWORDS = ["gun", "knife", "pistol", "rifle", "weapon", "sword", "blade"]
SUSPICIOUS_OBJECTS = ["backpack", "suitcase", "handbag", "briefcase"]

ALERT_SEVERITIES = ["low", "medium", "high", "critical"]
THREAT_LEVELS = ["none", "low", "medium", "high", "critical"]
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
GAIT_THREAT_LABELS = ["normal", "suspicious", "aggressive", "running", "limping"]
AUDIO_THREAT_TYPES = ["gunshot", "scream", "explosion", "glass_break", "alarm", "fighting"]
BEHAVIOR_PATTERNS = ["loitering", "running", "fighting", "crowd_forming", "restricted_intrusion", "tailgating", "abandoned_object"]
DETECTION_TYPES = ["person", "vehicle", "weapon", "audio", "behavior", "crowd", "pose", "plate"]

DB_POOL_MIN = 2
DB_POOL_MAX = 12

db_pool: ThreadedConnectionPool = None
redis_client: aioredis.Redis = None
r2_client = None
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

yolo_detector: YOLO = None
yolo_pose_detector: YOLO = None
yolo_segmentor: YOLO = None
face_analyzer: FaceAnalysis = None
ocr_engine: PaddleOCR = None

active_sessions: Dict[str, Dict] = {}
active_ws_connections: Dict[str, WebSocket] = {}
audio_ws_connections: Dict[str, WebSocket] = {}

frame_detection_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
track_frame_counts: Dict[int, int] = defaultdict(int)
track_speed_histories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_SPEED_HISTORY))
track_position_histories: Dict[int, List] = defaultdict(list)
track_first_seen: Dict[int, float] = {}
track_last_seen: Dict[int, float] = {}
loiter_tracker: Dict[str, float] = {}
gait_signature_store: Dict[str, List] = defaultdict(list)
attention_heatmap: Dict[str, np.ndarray] = {}
crowd_density_grid: Dict[str, np.ndarray] = {}
behavior_event_log: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
audio_event_buffer: deque = deque(maxlen=MAX_AUDIO_BUFFER)
fight_motion_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
known_face_embeddings: List[Dict] = []
known_plate_registry: Dict[str, Dict] = {}
flagged_faces_cache: Dict[str, Dict] = {}
session_metrics: Dict[str, Dict] = defaultdict(dict)
demographic_counters: Dict[str, Counter] = defaultdict(Counter)
zone_intrusion_log: Dict[str, List] = defaultdict(list)
object_abandonment_tracker: Dict[str, Dict] = {}
night_mode_active: Dict[str, bool] = defaultdict(bool)
motion_background_subtractors: Dict[str, Any] = {}
optical_flow_prev: Dict[str, np.ndarray] = {}

FULL_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS persons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT,
    alias TEXT,
    embedding vector(512),
    face_quality_score FLOAT DEFAULT 0.0,
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    seen_count INTEGER DEFAULT 1,
    age_estimate INTEGER,
    age_range TEXT,
    gender TEXT,
    gender_confidence FLOAT,
    ethnicity_estimate TEXT,
    dominant_emotion TEXT,
    emotion_history JSONB DEFAULT '[]',
    gait_signature JSONB DEFAULT '[]',
    threat_level TEXT DEFAULT 'none',
    threat_score FLOAT DEFAULT 0.0,
    flagged BOOLEAN DEFAULT FALSE,
    flag_reason TEXT,
    flag_timestamp TIMESTAMPTZ,
    flag_operator TEXT,
    nationality TEXT,
    height_estimate_cm FLOAT,
    weight_estimate_kg FLOAT,
    hair_color TEXT,
    clothing_color TEXT,
    clothing_description TEXT,
    tattoo_detected BOOLEAN DEFAULT FALSE,
    glasses_detected BOOLEAN DEFAULT FALSE,
    mask_detected BOOLEAN DEFAULT FALSE,
    known_associates JSONB DEFAULT '[]',
    location_log JSONB DEFAULT '[]',
    speed_history JSONB DEFAULT '[]',
    behavior_tags JSONB DEFAULT '[]',
    snapshot_url TEXT,
    additional_snapshots JSONB DEFAULT '[]',
    notes TEXT,
    created_by TEXT DEFAULT 'system',
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS vehicles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plate_number TEXT,
    plate_confidence FLOAT,
    plate_region TEXT,
    vehicle_type TEXT,
    vehicle_make TEXT,
    vehicle_model TEXT,
    vehicle_color TEXT,
    vehicle_color_hex TEXT,
    year_estimate INTEGER,
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    seen_count INTEGER DEFAULT 1,
    avg_speed_kph FLOAT DEFAULT 0.0,
    max_speed_kph FLOAT DEFAULT 0.0,
    speed_history JSONB DEFAULT '[]',
    route_history JSONB DEFAULT '[]',
    flagged BOOLEAN DEFAULT FALSE,
    flag_reason TEXT,
    flag_timestamp TIMESTAMPTZ,
    stolen BOOLEAN DEFAULT FALSE,
    owner_person_id UUID REFERENCES persons(id) ON DELETE SET NULL,
    linked_persons JSONB DEFAULT '[]',
    snapshot_url TEXT,
    additional_snapshots JSONB DEFAULT '[]',
    damage_detected BOOLEAN DEFAULT FALSE,
    notes TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT NOT NULL,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    detection_type TEXT NOT NULL,
    sub_type TEXT,
    person_id UUID REFERENCES persons(id) ON DELETE SET NULL,
    vehicle_id UUID REFERENCES vehicles(id) ON DELETE SET NULL,
    track_id INTEGER,
    confidence FLOAT,
    threat_score FLOAT DEFAULT 0.0,
    emotion TEXT,
    emotion_scores JSONB,
    age_estimate INTEGER,
    gender TEXT,
    pose_threat BOOLEAN DEFAULT FALSE,
    pose_reasons JSONB DEFAULT '[]',
    speed_kph FLOAT,
    plate_number TEXT,
    audio_type TEXT,
    bbox JSONB,
    keypoints JSONB,
    frame_number INTEGER,
    frame_snapshot_url TEXT,
    location JSONB DEFAULT '{}',
    is_returning BOOLEAN DEFAULT FALSE,
    is_flagged BOOLEAN DEFAULT FALSE,
    behavior_flags JSONB DEFAULT '[]',
    night_mode BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'low',
    priority INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolved_by TEXT,
    resolution_note TEXT,
    detection_id UUID REFERENCES detections(id) ON DELETE SET NULL,
    person_id UUID REFERENCES persons(id) ON DELETE SET NULL,
    vehicle_id UUID REFERENCES vehicles(id) ON DELETE SET NULL,
    session_id TEXT,
    description TEXT,
    snapshot_url TEXT,
    auto_generated BOOLEAN DEFAULT TRUE,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT,
    escalated BOOLEAN DEFAULT FALSE,
    escalation_level INTEGER DEFAULT 0,
    related_alert_ids JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS audio_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    event_type TEXT NOT NULL,
    confidence FLOAT,
    energy_level FLOAT,
    frequency_peak FLOAT,
    duration_ms FLOAT,
    session_id TEXT,
    audio_url TEXT,
    waveform_data JSONB DEFAULT '[]',
    spectral_features JSONB DEFAULT '{}',
    alert_triggered BOOLEAN DEFAULT FALSE,
    alert_id UUID REFERENCES alerts(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    duration_seconds FLOAT,
    source_type TEXT,
    source_url TEXT,
    camera_index INTEGER,
    resolution_w INTEGER,
    resolution_h INTEGER,
    avg_fps FLOAT,
    total_frames INTEGER DEFAULT 0,
    processed_frames INTEGER DEFAULT 0,
    total_persons_unique INTEGER DEFAULT 0,
    total_vehicles_unique INTEGER DEFAULT 0,
    total_alerts INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0,
    critical_alerts INTEGER DEFAULT 0,
    high_alerts INTEGER DEFAULT 0,
    night_mode_used BOOLEAN DEFAULT FALSE,
    audio_enabled BOOLEAN DEFAULT FALSE,
    restricted_zones JSONB DEFAULT '[]',
    perspective_config JSONB DEFAULT '{}',
    operator TEXT DEFAULT 'system',
    notes TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_used TIMESTAMPTZ,
    request_count INTEGER DEFAULT 0,
    active BOOLEAN DEFAULT TRUE,
    permissions JSONB DEFAULT '["read","write"]',
    rate_limit_per_hour INTEGER DEFAULT 1000,
    allowed_ips JSONB DEFAULT '[]',
    created_by TEXT DEFAULT 'master',
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS operators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'analyst',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    active BOOLEAN DEFAULT TRUE,
    permissions JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS watchlist (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    list_name TEXT NOT NULL,
    list_type TEXT NOT NULL,
    person_id UUID REFERENCES persons(id) ON DELETE CASCADE,
    vehicle_plate TEXT,
    reason TEXT,
    priority INTEGER DEFAULT 1,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    added_by TEXT,
    expires_at TIMESTAMPTZ,
    active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS restricted_zones (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    zone_name TEXT NOT NULL,
    session_id TEXT,
    coordinates JSONB NOT NULL,
    zone_type TEXT DEFAULT 'exclusion',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    alert_severity TEXT DEFAULT 'high',
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS behavior_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    behavior_type TEXT NOT NULL,
    person_id UUID REFERENCES persons(id) ON DELETE SET NULL,
    vehicle_id UUID REFERENCES vehicles(id) ON DELETE SET NULL,
    track_id INTEGER,
    duration_seconds FLOAT,
    confidence FLOAT,
    location JSONB DEFAULT '{}',
    alert_generated BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS heatmap_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT,
    captured_at TIMESTAMPTZ DEFAULT NOW(),
    heatmap_type TEXT,
    grid_data JSONB,
    image_url TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS demographic_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    period_start TIMESTAMPTZ,
    period_end TIMESTAMPTZ,
    total_persons INTEGER,
    gender_breakdown JSONB,
    age_breakdown JSONB,
    emotion_breakdown JSONB,
    threat_level_breakdown JSONB,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS plate_reads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT,
    read_at TIMESTAMPTZ DEFAULT NOW(),
    plate_number TEXT,
    confidence FLOAT,
    vehicle_id UUID REFERENCES vehicles(id) ON DELETE SET NULL,
    speed_at_read FLOAT,
    direction TEXT,
    lane INTEGER,
    snapshot_url TEXT,
    flagged BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS gait_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES persons(id) ON DELETE CASCADE,
    signature JSONB NOT NULL,
    confidence FLOAT,
    captured_at TIMESTAMPTZ DEFAULT NOW(),
    session_id TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    logged_at TIMESTAMPTZ DEFAULT NOW(),
    level TEXT,
    component TEXT,
    message TEXT,
    session_id TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_persons_embedding ON persons USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_persons_flagged ON persons(flagged);
CREATE INDEX IF NOT EXISTS idx_persons_threat_level ON persons(threat_level);
CREATE INDEX IF NOT EXISTS idx_persons_last_seen ON persons(last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_detections_session ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(detection_type);
CREATE INDEX IF NOT EXISTS idx_detections_at ON detections(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_person ON detections(person_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles(plate_number);
CREATE INDEX IF NOT EXISTS idx_vehicles_flagged ON vehicles(flagged);
CREATE INDEX IF NOT EXISTS idx_plate_reads_plate ON plate_reads(plate_number);
CREATE INDEX IF NOT EXISTS idx_plate_reads_at ON plate_reads(read_at DESC);
CREATE INDEX IF NOT EXISTS idx_behavior_session ON behavior_events(session_id);
CREATE INDEX IF NOT EXISTS idx_behavior_type ON behavior_events(behavior_type);
CREATE INDEX IF NOT EXISTS idx_watchlist_active ON watchlist(active);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_system_logs_at ON system_logs(logged_at DESC);
"""

@dataclass
class DetectionResult:
    det_type: str
    bbox: List[float]
    confidence: float
    track_id: int = -1
    person_id: Optional[str] = None
    vehicle_id: Optional[str] = None
    name: Optional[str] = None
    is_returning: bool = False
    is_flagged: bool = False
    threat_level: str = "none"
    threat_score: float = 0.0
    emotion: Optional[str] = None
    emotion_scores: Optional[Dict] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    speed_kph: Optional[float] = None
    plate: Optional[str] = None
    plate_confidence: Optional[float] = None
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    pose_threat: bool = False
    pose_reasons: List[str] = field(default_factory=list)
    keypoints: Optional[List] = None
    behavior_flags: List[str] = field(default_factory=list)
    gait_label: Optional[str] = None
    mask_detected: bool = False
    glasses_detected: bool = False
    night_mode: bool = False
    snapshot_url: Optional[str] = None
    sub_type: Optional[str] = None
    audio_type: Optional[str] = None
    flag_reason: Optional[str] = None

@dataclass
class AlertResult:
    alert_type: str
    severity: str
    description: str
    person_id: Optional[str] = None
    vehicle_id: Optional[str] = None
    track_id: int = -1
    plate: Optional[str] = None
    speed_kph: Optional[float] = None
    snapshot_url: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

def db_connect():
    global db_pool
    db_pool = ThreadedConnectionPool(DB_POOL_MIN, DB_POOL_MAX, DB_URL)

def get_db_conn():
    return db_pool.getconn()

def release_db_conn(conn):
    db_pool.putconn(conn)

def db_execute(query: str, params: tuple = None, fetch: str = None):
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            conn.commit()
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            if fetch == "scalar":
                row = cur.fetchone()
                return list(row.values())[0] if row else None
    except Exception as e:
        conn.rollback()
        logger.error(f"DB error: {e}")
        raise
    finally:
        release_db_conn(conn)

def init_database():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(FULL_SCHEMA)
            master_hash = hashlib.sha256(MASTER_API_KEY.encode()).hexdigest()
            cur.execute(
                "INSERT INTO api_keys (key_hash, name, permissions) VALUES (%s, %s, %s::jsonb) ON CONFLICT (key_hash) DO NOTHING",
                (master_hash, "master", json.dumps(["read", "write", "admin", "delete"]))
            )
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Database init failed: {e}")
        raise
    finally:
        release_db_conn(conn)

def init_redis():
    global redis_client
    redis_client = aioredis.from_url(
        UPSTASH_REDIS_URL,
        password=UPSTASH_REDIS_TOKEN,
        decode_responses=True,
        ssl=True
    )

def init_r2():
    global r2_client
    r2_client = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def init_ai_models():
    global yolo_detector, yolo_pose_detector, yolo_segmentor, face_analyzer, ocr_engine
    logger.info("Loading YOLO detector...")
    yolo_detector = YOLO("yolov8x.pt")
    logger.info("Loading YOLO pose...")
    yolo_pose_detector = YOLO("yolov8x-pose.pt")
    logger.info("Loading YOLO segmentor...")
    yolo_segmentor = YOLO("yolov8x-seg.pt")
    logger.info("Loading InsightFace...")
    face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("Loading PaddleOCR...")
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
    logger.info("All AI models loaded")

def load_known_faces_from_db():
    global known_face_embeddings, flagged_faces_cache
    rows = db_execute(
        "SELECT id, name, alias, embedding, flagged, flag_reason, threat_level, threat_score, age_estimate, gender FROM persons WHERE embedding IS NOT NULL",
        fetch="all"
    )
    known_face_embeddings = []
    flagged_faces_cache = {}
    for row in (rows or []):
        row = dict(row)
        emb = row["embedding"]
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except Exception:
                continue
        arr = np.array(emb, dtype=np.float32)
        entry = {
            "id": str(row["id"]),
            "name": row.get("name"),
            "alias": row.get("alias"),
            "embedding": arr,
            "flagged": row.get("flagged", False),
            "flag_reason": row.get("flag_reason"),
            "threat_level": row.get("threat_level", "none"),
            "threat_score": row.get("threat_score", 0.0),
            "age_estimate": row.get("age_estimate"),
            "gender": row.get("gender"),
        }
        known_face_embeddings.append(entry)
        if row.get("flagged"):
            flagged_faces_cache[str(row["id"])] = entry
    logger.info(f"Loaded {len(known_face_embeddings)} known faces ({len(flagged_faces_cache)} flagged)")

def load_known_plates_from_db():
    global known_plate_registry
    rows = db_execute("SELECT id, plate_number, flagged, flag_reason, stolen, seen_count, vehicle_type, vehicle_color, owner_person_id FROM vehicles WHERE plate_number IS NOT NULL", fetch="all")
    known_plate_registry = {}
    for row in (rows or []):
        row = dict(row)
        known_plate_registry[row["plate_number"]] = row
    logger.info(f"Loaded {len(known_plate_registry)} known plates")

def upload_bytes_to_r2(data: bytes, key: str, content_type: str = "image/jpeg") -> str:
    try:
        r2_client.put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return f"https://{R2_BUCKET}.{R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{key}"
    except Exception as e:
        logger.error(f"R2 upload failed: {e}")
        return ""

def upload_frame_snapshot(frame: np.ndarray, prefix: str = "snapshots") -> str:
    try:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])
        key = f"{prefix}/{datetime.datetime.utcnow().strftime('%Y/%m/%d')}/{uuid.uuid4().hex}.jpg"
        return upload_bytes_to_r2(buf.tobytes(), key)
    except Exception as e:
        logger.error(f"Frame snapshot upload failed: {e}")
        return ""

def frame_to_b64(frame: np.ndarray, quality: int = 75) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()

def crop_region(frame: np.ndarray, bbox: List[float], pad: int = 8) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(w, int(bbox[2]) + pad)
    y2 = min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]

def enhance_for_night(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.detailEnhance(enhanced, sigma_s=8, sigma_r=0.12)
    return enhanced

def detect_brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def is_night_mode(frame: np.ndarray) -> bool:
    return detect_brightness(frame) < NIGHT_MODE_THRESHOLD

def compute_dominant_color(crop: np.ndarray) -> Tuple[str, str]:
    try:
        small = cv2.resize(crop, (50, 50))
        pixels = small.reshape(-1, 3).astype(np.float32)
        _, labels, centers = cv2.kmeans(pixels, 3, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            3, cv2.KMEANS_RANDOM_CENTERS)
        counts = Counter(labels.flatten())
        dominant = centers[counts.most_common(1)[0][0]]
        b, g, r = int(dominant[0]), int(dominant[1]), int(dominant[2])
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        color_name = bgr_to_color_name(b, g, r)
        return color_name, hex_color
    except Exception:
        return "unknown", "#000000"

def bgr_to_color_name(b: int, g: int, r: int) -> str:
    colors = {
        "white": (255, 255, 255), "black": (0, 0, 0), "red": (0, 0, 255),
        "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255),
        "orange": (0, 165, 255), "purple": (128, 0, 128), "gray": (128, 128, 128),
        "silver": (192, 192, 192), "brown": (42, 42, 165), "cyan": (255, 255, 0),
    }
    min_dist = float("inf")
    best = "unknown"
    for name, (cb, cg, cr) in colors.items():
        d = math.sqrt((b - cb) ** 2 + (g - cg) ** 2 + (r - cr) ** 2)
        if d < min_dist:
            min_dist = d
            best = name
    return best

def match_face_embedding(embedding: np.ndarray) -> Optional[Dict]:
    if not known_face_embeddings:
        return None
    best_score = float("inf")
    best_match = None
    for known in known_face_embeddings:
        try:
            score = cosine(embedding, known["embedding"])
            if score < best_score:
                best_score = score
                best_match = known
        except Exception:
            continue
    if best_match and best_score < FACE_SIMILARITY_THRESHOLD:
        return {**best_match, "similarity_score": round(1 - best_score, 4), "distance": round(best_score, 4)}
    return None

def detect_faces_on_crop(crop: np.ndarray) -> List[Dict]:
    try:
        faces = face_analyzer.get(crop)
        results = []
        for f in faces:
            det_score = float(f.det_score) if hasattr(f, "det_score") else 0.0
            if det_score < FACE_DETECTION_CONFIDENCE:
                continue
            age = int(f.age) if hasattr(f, "age") and f.age is not None else None
            gender_raw = f.gender if hasattr(f, "gender") else None
            gender = "M" if gender_raw == 1 else "F" if gender_raw == 0 else "U"
            results.append({
                "embedding": f.embedding,
                "bbox": f.bbox.tolist() if hasattr(f, "bbox") else [],
                "det_score": det_score,
                "age": age,
                "gender": gender,
                "kps": f.kps.tolist() if hasattr(f, "kps") and f.kps is not None else [],
            })
        return results
    except Exception as e:
        logger.debug(f"Face detection error: {e}")
        return []

def analyze_emotion_on_crop(crop: np.ndarray) -> Dict:
    try:
        result = DeepFace.analyze(
            crop, actions=["emotion"],
            enforce_detection=False, silent=True
        )
        if isinstance(result, list):
            result = result[0]
        scores = result.get("emotion", {})
        dominant = result.get("dominant_emotion", "neutral")
        return {"dominant": dominant, "scores": scores}
    except Exception:
        return {"dominant": "neutral", "scores": {e: 0.0 for e in EMOTION_LABELS}}

def detect_accessories(crop: np.ndarray) -> Dict:
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        upper_region = gray[:h // 2, :]
        brightness = np.mean(upper_region)
        mask_detected = False
        glasses_detected = False
        lower_region = gray[h // 2:, :]
        lower_edges = cv2.Canny(lower_region, 50, 150)
        if np.sum(lower_edges > 0) > (lower_region.size * 0.15):
            mask_detected = True
        horizontal_edges = cv2.Sobel(upper_region, cv2.CV_64F, 1, 0, ksize=3)
        if np.sum(np.abs(horizontal_edges) > 30) > (upper_region.size * 0.08):
            glasses_detected = True
        return {"mask": mask_detected, "glasses": glasses_detected}
    except Exception:
        return {"mask": False, "glasses": False}

def estimate_height(bbox: List[float], frame_h: int, camera_height_m: float = 3.0) -> float:
    try:
        pixel_height = bbox[3] - bbox[1]
        focal_length = frame_h * 0.8
        height_m = (camera_height_m * pixel_height) / focal_length
        return round(height_m * 100, 1)
    except Exception:
        return 0.0

def analyze_pose_keypoints(keypoints: np.ndarray) -> Dict:
    threat = False
    reasons = []
    gait_features = []
    try:
        if keypoints.shape[0] < 17:
            return {"threat": False, "reasons": [], "gait_features": []}
        nose = keypoints[0][:2]
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_elbow = keypoints[7][:2]
        right_elbow = keypoints[8][:2]
        left_wrist = keypoints[9][:2]
        right_wrist = keypoints[10][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]
        left_ankle = keypoints[15][:2]
        right_ankle = keypoints[16][:2]

        def compute_angle(a, b, c):
            v1 = a - b
            v2 = c - b
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                return 0.0
            cos_val = np.dot(v1, v2) / (n1 * n2)
            return float(np.degrees(np.arccos(np.clip(cos_val, -1, 1))))

        left_elbow_angle = compute_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = compute_angle(right_shoulder, right_elbow, right_wrist)
        left_knee_angle = compute_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = compute_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = compute_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = compute_angle(right_shoulder, right_hip, right_knee)

        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2

        if left_wrist[1] < shoulder_mid[1] or right_wrist[1] < shoulder_mid[1]:
            threat = True
            reasons.append("hands_above_shoulders")

        if left_elbow_angle > 155 and right_elbow_angle > 155:
            threat = True
            reasons.append("both_arms_fully_extended")

        torso_lean = abs(shoulder_mid[0] - hip_mid[0])
        torso_height = abs(shoulder_mid[1] - hip_mid[1])
        if torso_height > 0 and torso_lean / torso_height > 0.45:
            threat = True
            reasons.append("aggressive_lean_forward")

        wrist_separation = np.linalg.norm(left_wrist - right_wrist)
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_width > 0 and wrist_separation / shoulder_width > 2.5:
            reasons.append("wide_arm_spread")

        step_width = abs(left_ankle[0] - right_ankle[0])
        step_height_diff = abs(left_ankle[1] - right_ankle[1])
        gait_features = [
            float(left_knee_angle), float(right_knee_angle),
            float(left_hip_angle), float(right_hip_angle),
            float(left_elbow_angle), float(right_elbow_angle),
            float(step_width), float(step_height_diff),
            float(torso_lean), float(torso_height),
        ]
    except Exception as e:
        logger.debug(f"Pose analysis error: {e}")

    return {"threat": threat, "reasons": reasons, "gait_features": gait_features}

def classify_gait(gait_history: List[List[float]]) -> str:
    if len(gait_history) < 5:
        return "normal"
    try:
        arr = np.array(gait_history[-10:])
        step_variance = np.var(arr[:, 6])
        knee_angles = arr[:, 0]
        avg_knee = np.mean(knee_angles)
        knee_change_rate = np.mean(np.abs(np.diff(knee_angles)))
        if knee_change_rate > 25:
            return "running"
        if step_variance > 500:
            return "suspicious"
        if avg_knee < 140:
            return "aggressive"
        return "normal"
    except Exception:
        return "normal"

def compute_optical_flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(magnitude))
    except Exception:
        return 0.0

def detect_fight_motion(session_id: str, flow_magnitude: float) -> bool:
    fight_motion_buffer[session_id].append(flow_magnitude)
    if len(fight_motion_buffer[session_id]) < 10:
        return False
    recent = list(fight_motion_buffer[session_id])[-10:]
    avg_flow = np.mean(recent)
    variance = np.var(recent)
    return float(avg_flow) > FIGHT_MOTION_THRESHOLD and float(variance) > 200.0

def run_plate_ocr(crop: np.ndarray) -> Tuple[str, float]:
    try:
        result = ocr_engine.ocr(crop, cls=True)
        if not result or not result[0]:
            return "", 0.0
        best_text = ""
        best_conf = 0.0
        for line in result[0]:
            text = line[1][0]
            conf = float(line[1][1])
            cleaned = "".join(c for c in text.upper() if c.isalnum())
            if PLATE_MIN_CHARS <= len(cleaned) <= PLATE_MAX_CHARS and conf > best_conf:
                best_conf = conf
                best_text = cleaned
        return best_text, best_conf
    except Exception as e:
        logger.debug(f"OCR error: {e}")
        return "", 0.0

def preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(resized, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(denoised)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    except Exception:
        return crop

def compute_vehicle_speed(track_id: int, positions: List, fps: float, perspective_matrix: Optional[np.ndarray]) -> float:
    if len(positions) < 2 or perspective_matrix is None:
        return 0.0
    try:
        pts = np.array([[p] for p in positions[-2:]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, perspective_matrix)
        dist_m = np.linalg.norm(warped[1][0] - warped[0][0])
        time_s = 1.0 / max(fps, 1.0)
        speed_kph = (dist_m / time_s) * 3.6
        if MIN_PLAUSIBLE_KPH < speed_kph < MAX_PLAUSIBLE_KPH:
            track_speed_histories[track_id].append(speed_kph)
            return round(float(np.mean(list(track_speed_histories[track_id]))), 1)
        return 0.0
    except Exception:
        return 0.0

def classify_audio_chunk(audio_data: np.ndarray, sr: int) -> Dict:
    try:
        if len(audio_data) == 0:
            return {"threat": False, "type": None, "confidence": 0.0}
        audio_f = audio_data.astype(np.float32)
        if np.max(np.abs(audio_f)) > 0:
            audio_f = audio_f / np.max(np.abs(audio_f))
        energy = float(np.sum(audio_f ** 2))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_f)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio_f, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio_f, sr=sr)))
        mfcc = librosa.feature.mfcc(y=audio_f, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        rms = float(np.sqrt(np.mean(audio_f ** 2)))
        if energy > 0.8 and zcr > 0.12 and centroid > 4000 and rms > 0.3:
            return {"threat": True, "type": "gunshot", "confidence": min(0.95, energy), "energy": energy, "centroid": centroid}
        if rms > 0.15 and centroid > 2500 and zcr > 0.08:
            return {"threat": True, "type": "scream", "confidence": 0.72, "energy": energy, "centroid": centroid}
        if energy > 1.2 and centroid < 2000 and rolloff > 8000:
            return {"threat": True, "type": "explosion", "confidence": 0.68, "energy": energy, "centroid": centroid}
        if zcr > 0.2 and centroid > 5000 and rms > 0.1:
            return {"threat": True, "type": "glass_break", "confidence": 0.65, "energy": energy, "centroid": centroid}
        if rms > 0.1 and 0.05 < zcr < 0.15 and centroid > 1500:
            return {"threat": True, "type": "fighting", "confidence": 0.55, "energy": energy, "centroid": centroid}
        return {"threat": False, "type": None, "confidence": 0.0, "energy": energy, "centroid": centroid}
    except Exception as e:
        logger.debug(f"Audio classification error: {e}")
        return {"threat": False, "type": None, "confidence": 0.0}

def update_attention_heatmap(session_id: str, frame_shape: Tuple, detections: List[DetectionResult]):
    h, w = frame_shape[:2]
    grid_h = h // HEATMAP_GRID_SIZE
    grid_w = w // HEATMAP_GRID_SIZE
    if session_id not in attention_heatmap:
        attention_heatmap[session_id] = np.zeros((grid_h, grid_w), dtype=np.float32)
    hm = attention_heatmap[session_id]
    hm *= ATTENTION_HEATMAP_DECAY
    for det in detections:
        cx = (det.bbox[0] + det.bbox[2]) / 2
        cy = (det.bbox[1] + det.bbox[3]) / 2
        gx = min(int(cx / HEATMAP_GRID_SIZE), grid_w - 1)
        gy = min(int(cy / HEATMAP_GRID_SIZE), grid_h - 1)
        hm[gy, gx] += 1.0
    attention_heatmap[session_id] = hm

def render_heatmap_overlay(frame: np.ndarray, session_id: str) -> np.ndarray:
    if session_id not in attention_heatmap:
        return frame
    hm = attention_heatmap[session_id]
    h, w = frame.shape[:2]
    hm_resized = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    if np.max(hm_resized) > 0:
        hm_norm = (hm_resized / np.max(hm_resized) * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.75, hm_color, 0.25, 0)
    return frame

def update_crowd_density(session_id: str, frame_shape: Tuple, person_positions: List[Tuple]):
    h, w = frame_shape[:2]
    grid_h = max(1, h // HEATMAP_GRID_SIZE)
    grid_w = max(1, w // HEATMAP_GRID_SIZE)
    grid = np.zeros((grid_h, grid_w), dtype=np.int32)
    for cx, cy in person_positions:
        gx = min(int(cx / HEATMAP_GRID_SIZE), grid_w - 1)
        gy = min(int(cy / HEATMAP_GRID_SIZE), grid_h - 1)
        grid[gy, gx] += 1
    crowd_density_grid[session_id] = grid

def detect_abandoned_object(session_id: str, track_id: int, det_type: str, position: Tuple, time_now: float) -> bool:
    if det_type not in SUSPICIOUS_OBJECTS:
        return False
    key = f"{session_id}_{track_id}"
    if key not in object_abandonment_tracker:
        object_abandonment_tracker[key] = {"first_seen": time_now, "position": position, "det_type": det_type}
        return False
    entry = object_abandonment_tracker[key]
    dist = math.sqrt((position[0] - entry["position"][0]) ** 2 + (position[1] - entry["position"][1]) ** 2)
    if dist < 30 and (time_now - entry["first_seen"]) > 45.0:
        return True
    return False

def detect_tailgating(session_id: str, vehicle_tracks: List[Dict]) -> List[Dict]:
    alerts = []
    if len(vehicle_tracks) < 2:
        return alerts
    for i in range(len(vehicle_tracks)):
        for j in range(i + 1, len(vehicle_tracks)):
            v1 = vehicle_tracks[i]
            v2 = vehicle_tracks[j]
            dist = math.sqrt(
                (v1["cx"] - v2["cx"]) ** 2 +
                (v1["cy"] - v2["cy"]) ** 2
            )
            speed_diff = abs((v1.get("speed", 0) or 0) - (v2.get("speed", 0) or 0))
            if dist < 80 and speed_diff < 20:
                alerts.append({
                    "track_a": v1["track_id"],
                    "track_b": v2["track_id"],
                    "distance_px": round(dist, 1),
                })
    return alerts

def compute_age_range(age: int) -> str:
    for lo, hi in DEMOGRAPHIC_AGE_BINS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "unknown"

def db_save_person(embedding: np.ndarray, age: int, gender: str, gender_conf: float,
                    snapshot_url: str, age_range: str, accessories: Dict, height_cm: float,
                    clothing_color: str, session_id: str) -> str:
    person_id = str(uuid.uuid4())
    emb_list = embedding.tolist()
    db_execute(
        """INSERT INTO persons (id, embedding, age_estimate, age_range, gender, gender_confidence,
           snapshot_url, glasses_detected, mask_detected, height_estimate_cm, clothing_color, created_by)
           VALUES (%s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (person_id, json.dumps(emb_list), age, age_range, gender, gender_conf,
         snapshot_url, accessories.get("glasses", False), accessories.get("mask", False),
         height_cm, clothing_color, session_id)
    )
    return person_id

def db_update_person_seen(person_id: str, location: Dict, emotion: str, session_id: str):
    db_execute(
        """UPDATE persons SET last_seen = NOW(), seen_count = seen_count + 1,
           location_log = location_log || %s::jsonb,
           dominant_emotion = %s,
           emotion_history = emotion_history || %s::jsonb
           WHERE id = %s""",
        (
            json.dumps([{**location, "session": session_id, "ts": datetime.datetime.utcnow().isoformat()}]),
            emotion,
            json.dumps([{"emotion": emotion, "ts": datetime.datetime.utcnow().isoformat()}]),
            person_id
        )
    )

def db_update_person_gait(person_id: str, gait_label: str, gait_features: List[float], session_id: str):
    db_execute(
        """UPDATE persons SET gait_signature = gait_signature || %s::jsonb WHERE id = %s""",
        (json.dumps([{"label": gait_label, "features": gait_features, "session": session_id}]), person_id)
    )
    db_execute(
        "INSERT INTO gait_profiles (person_id, signature, confidence, session_id) VALUES (%s, %s::jsonb, %s, %s)",
        (person_id, json.dumps(gait_features), 0.7, session_id)
    )

def db_save_vehicle(plate: str, plate_conf: float, vehicle_type: str, vehicle_color: str,
                     vehicle_color_hex: str, snapshot_url: str) -> str:
    vehicle_id = str(uuid.uuid4())
    db_execute(
        """INSERT INTO vehicles (id, plate_number, plate_confidence, vehicle_type, vehicle_color, vehicle_color_hex, snapshot_url)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (id) DO NOTHING""",
        (vehicle_id, plate, plate_conf, vehicle_type, vehicle_color, vehicle_color_hex, snapshot_url)
    )
    row = db_execute("SELECT id FROM vehicles WHERE plate_number = %s ORDER BY first_seen ASC LIMIT 1", (plate,), fetch="one")
    if row:
        vehicle_id = str(row["id"])
        db_execute("UPDATE vehicles SET last_seen = NOW(), seen_count = seen_count + 1 WHERE id = %s", (vehicle_id,))
    return vehicle_id

def db_save_plate_read(session_id: str, plate: str, confidence: float, vehicle_id: str,
                        speed: float, snapshot_url: str, flagged: bool):
    db_execute(
        """INSERT INTO plate_reads (session_id, plate_number, confidence, vehicle_id, speed_at_read, snapshot_url, flagged)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (session_id, plate, confidence, vehicle_id, speed, snapshot_url, flagged)
    )

def db_update_vehicle_speed(vehicle_id: str, speed_kph: float):
    db_execute(
        """UPDATE vehicles SET
           speed_history = speed_history || %s::jsonb,
           avg_speed_kph = (avg_speed_kph + %s) / 2,
           max_speed_kph = GREATEST(max_speed_kph, %s),
           last_seen = NOW()
           WHERE id = %s""",
        (json.dumps([{"speed": speed_kph, "ts": datetime.datetime.utcnow().isoformat()}]),
         speed_kph, speed_kph, vehicle_id)
    )

def db_save_detection(session_id: str, det: DetectionResult, frame_num: int, snap_url: str) -> str:
    det_id = str(uuid.uuid4())
    db_execute(
        """INSERT INTO detections (id, session_id, detection_type, sub_type, person_id, vehicle_id,
           track_id, confidence, threat_score, emotion, emotion_scores, age_estimate, gender,
           pose_threat, pose_reasons, speed_kph, plate_number, audio_type, bbox, frame_number,
           frame_snapshot_url, is_returning, is_flagged, behavior_flags, night_mode, metadata)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb)""",
        (det_id, session_id, det.det_type, det.sub_type, det.person_id, det.vehicle_id,
         det.track_id, det.confidence, det.threat_score, det.emotion,
         json.dumps(det.emotion_scores or {}), det.age, det.gender,
         det.pose_threat, json.dumps(det.pose_reasons), det.speed_kph,
         det.plate, det.audio_type, json.dumps(det.bbox), frame_num,
         snap_url, det.is_returning, det.is_flagged,
         json.dumps(det.behavior_flags), det.night_mode, json.dumps({}))
    )
    return det_id

def db_save_alert(alert: AlertResult, session_id: str, detection_id: Optional[str]) -> str:
    alert_id = str(uuid.uuid4())
    priority = {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(alert.severity, 1)
    db_execute(
        """INSERT INTO alerts (id, alert_type, severity, priority, detection_id, person_id, vehicle_id,
           session_id, description, snapshot_url, metadata)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)""",
        (alert_id, alert.alert_type, alert.severity, priority, detection_id,
         alert.person_id, alert.vehicle_id, session_id, alert.description,
         alert.snapshot_url, json.dumps(alert.metadata))
    )
    return alert_id

def db_save_audio_event(session_id: str, event_type: str, confidence: float,
                          energy: float, centroid: float) -> str:
    event_id = str(uuid.uuid4())
    db_execute(
        """INSERT INTO audio_events (id, event_type, confidence, energy_level, frequency_peak, session_id)
           VALUES (%s,%s,%s,%s,%s,%s)""",
        (event_id, event_type, confidence, energy, centroid, session_id)
    )
    return event_id

def db_save_behavior_event(session_id: str, behavior_type: str, person_id: Optional[str],
                             track_id: int, duration: float, confidence: float, location: Dict):
    db_execute(
        """INSERT INTO behavior_events (session_id, behavior_type, person_id, track_id, duration_seconds, confidence, location)
           VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)""",
        (session_id, behavior_type, person_id, track_id, duration, confidence, json.dumps(location))
    )

def db_save_session(session_id: str, source_type: str, source_url: str,
                     camera_index: int, audio_enabled: bool, restricted_zones: List,
                     perspective_config: Dict, operator: str):
    db_execute(
        """INSERT INTO sessions (id, source_type, source_url, camera_index, audio_enabled,
           restricted_zones, perspective_config, operator)
           VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s) ON CONFLICT (id) DO NOTHING""",
        (session_id, source_type, source_url, camera_index, audio_enabled,
         json.dumps(restricted_zones), json.dumps(perspective_config), operator)
    )

def db_update_session_stats(session_id: str, s: Dict):
    db_execute(
        """UPDATE sessions SET total_frames=%s, processed_frames=%s, total_persons_unique=%s,
           total_vehicles_unique=%s, total_alerts=%s, total_detections=%s, avg_fps=%s
           WHERE id=%s""",
        (s.get("frame_count", 0), s.get("processed_frames", 0), s.get("unique_persons", 0),
         s.get("unique_vehicles", 0), s.get("alert_count", 0), s.get("detection_count", 0),
         s.get("fps", 0.0), session_id)
    )

def db_end_session(session_id: str, duration_s: float):
    db_execute(
        "UPDATE sessions SET ended_at = NOW(), duration_seconds = %s WHERE id = %s",
        (duration_s, session_id)
    )

def db_log_system(level: str, component: str, message: str, session_id: Optional[str] = None):
    try:
        db_execute(
            "INSERT INTO system_logs (level, component, message, session_id) VALUES (%s,%s,%s,%s)",
            (level, component, message, session_id)
        )
    except Exception:
        pass

def check_api_key(key: str) -> Optional[Dict]:
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    row = db_execute(
        "SELECT id, name, active, permissions, rate_limit_per_hour FROM api_keys WHERE key_hash = %s",
        (key_hash,), fetch="one"
    )
    if row and row["active"]:
        db_execute(
            "UPDATE api_keys SET last_used = NOW(), request_count = request_count + 1 WHERE key_hash = %s",
            (key_hash,)
        )
        return dict(row)
    return None

def verify_jwt_token(token: str) -> Optional[Dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception:
        return None

security_scheme = HTTPBearer(auto_error=False)

async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    x_api_key: Optional[str] = Header(None),
    request: Request = None,
):
    if x_api_key:
        key_data = check_api_key(x_api_key)
        if key_data:
            return {"type": "api_key", "name": key_data["name"], "permissions": key_data["permissions"]}
    if credentials:
        payload = verify_jwt_token(credentials.credentials)
        if payload:
            return payload
    raise HTTPException(status_code=401, detail="Unauthorized — provide a valid API key or token")

async def require_admin(user=Depends(require_auth)):
    perms = user.get("permissions", [])
    if "admin" not in perms and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin permission required")
    return user

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 200, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self._store: Dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._store[ip]
        while window and now - window[0] > self.period:
            window.popleft()
        if len(window) >= self.calls:
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        window.append(now)
        return await call_next(request)

app = FastAPI(
    title="ARGUS Surveillance System",
    description="Full-spectrum AI surveillance: face recognition, vehicle tracking, behavior analysis, threat detection",
    version="2.0.0",
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url=None,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(RateLimitMiddleware, calls=300, period=60)

class SessionStartRequest(BaseModel):
    source_type: str = Field(..., description="webcam | rtsp | ip_cam | file | screen")
    source_url: Optional[str] = None
    camera_index: int = 0
    perspective_src: Optional[List[List[float]]] = None
    perspective_dst: Optional[List[List[float]]] = None
    enable_audio: bool = False
    enable_heatmap: bool = False
    enable_night_mode: bool = False
    restricted_zones: Optional[List[List[float]]] = []
    operator: str = "system"
    notes: Optional[str] = None

class PersonFlagRequest(BaseModel):
    person_id: str
    reason: str
    threat_level: str = "high"
    operator: str = "system"

class PersonNameRequest(BaseModel):
    person_id: str
    name: str
    alias: Optional[str] = None
    notes: Optional[str] = None

class PersonLinkRequest(BaseModel):
    person_id_a: str
    person_id_b: str
    relationship: str = "associate"

class VehicleFlagRequest(BaseModel):
    vehicle_id: str
    reason: str
    stolen: bool = False

class AlertResolveRequest(BaseModel):
    alert_id: str
    resolution_note: Optional[str] = None
    resolved_by: str = "operator"

class AlertAcknowledgeRequest(BaseModel):
    alert_id: str
    acknowledged_by: str = "operator"

class ApiKeyCreateRequest(BaseModel):
    name: str
    permissions: List[str] = ["read", "write"]
    rate_limit_per_hour: int = 1000
    expires_in_days: Optional[int] = None

class WatchlistAddRequest(BaseModel):
    list_name: str
    list_type: str = "persons"
    person_id: Optional[str] = None
    vehicle_plate: Optional[str] = None
    reason: str
    priority: int = 1
    added_by: str = "operator"

class ZoneCreateRequest(BaseModel):
    zone_name: str
    session_id: Optional[str] = None
    coordinates: List[float]
    zone_type: str = "exclusion"
    alert_severity: str = "high"

class OperatorCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "analyst"
    permissions: List[str] = []

def process_full_frame(
    frame: np.ndarray,
    session_id: str,
    frame_count: int,
    fps: float,
    perspective_matrix: Optional[np.ndarray],
    restricted_zones: List,
    enable_heatmap: bool,
    enable_night_mode: bool,
) -> Tuple[List[DetectionResult], List[AlertResult]]:

    detections: List[DetectionResult] = []
    alerts: List[AlertResult] = []
    time_now = time.time()
    h, w = frame.shape[:2]

    night_active = is_night_mode(frame)
    if night_active or enable_night_mode:
        frame = enhance_for_night(frame)
        night_mode_active[session_id] = True
    else:
        night_mode_active.pop(session_id, None)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = optical_flow_prev.get(session_id)
    flow_mag = 0.0
    if prev_gray is not None and prev_gray.shape == gray.shape:
        flow_mag = compute_optical_flow_magnitude(prev_gray, gray)
    optical_flow_prev[session_id] = gray.copy()

    if session_id not in motion_background_subtractors:
        motion_background_subtractors[session_id] = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )
    bg_sub = motion_background_subtractors[session_id]
    motion_mask = bg_sub.apply(frame)

    yolo_results = yolo_detector(frame, verbose=False)[0]
    pose_results = yolo_pose_detector(frame, verbose=False)[0]

    person_positions_for_crowd = []
    vehicle_tracks_for_tailgate = []

    boxes = yolo_results.boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            track_id = int(box.id[0]) if box.id is not None else -1
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            crop = crop_region(frame, xyxy)

            if cls_id == PERSON_CLASS and conf > PERSON_CONFIDENCE_THRESHOLD:
                person_positions_for_crowd.append((cx, cy))

                if track_id >= 0:
                    track_frame_counts[track_id] += 1
                    track_position_histories[track_id].append([cx, cy])
                    if len(track_position_histories[track_id]) > HISTORY_LEN:
                        track_position_histories[track_id].pop(0)
                    if track_id not in track_first_seen:
                        track_first_seen[track_id] = time_now
                    track_last_seen[track_id] = time_now

                det = DetectionResult(
                    det_type="person",
                    bbox=xyxy,
                    confidence=conf,
                    track_id=track_id,
                    night_mode=night_active,
                )

                face_data = detect_faces_on_crop(crop)
                if face_data:
                    fd = face_data[0]
                    embedding = fd["embedding"]
                    age = fd["age"] or 0
                    gender = fd["gender"]
                    gender_conf = fd["det_score"]

                    accessories = detect_accessories(crop)
                    det.mask_detected = accessories["mask"]
                    det.glasses_detected = accessories["glasses"]
                    det.age = age
                    det.gender = gender
                    det.gender_confidence = gender_conf

                    clothing_color, clothing_hex = compute_dominant_color(crop[int(crop.shape[0] * 0.4):])
                    det.vehicle_color = clothing_color

                    height_cm = estimate_height(xyxy, h)
                    age_range = compute_age_range(age) if age > 0 else "unknown"

                    if frame_count % EMOTION_INTERVAL_FRAMES == 0:
                        emotion_result = analyze_emotion_on_crop(crop)
                        det.emotion = emotion_result["dominant"]
                        det.emotion_scores = emotion_result["scores"]

                    match = match_face_embedding(embedding)

                    if match:
                        det.person_id = match["id"]
                        det.name = match.get("name")
                        det.is_returning = True
                        det.is_flagged = match.get("flagged", False)
                        det.flag_reason = match.get("flag_reason")
                        det.threat_level = match.get("threat_level", "none")
                        det.threat_score = match.get("threat_score", 0.0)

                        db_update_person_seen(
                            match["id"],
                            {"cx": cx, "cy": cy, "w": w, "h": h},
                            det.emotion or "neutral",
                            session_id
                        )

                        if match.get("flagged"):
                            snap_url = upload_frame_snapshot(crop, f"snapshots/flagged/{session_id}")
                            det.snapshot_url = snap_url
                            a = AlertResult(
                                alert_type="flagged_person_detected",
                                severity="critical",
                                description=f"Known flagged person detected. Reason: {match.get('flag_reason', 'N/A')}",
                                person_id=match["id"],
                                snapshot_url=snap_url,
                                metadata={"similarity": match.get("similarity_score"), "name": match.get("name")}
                            )
                            alerts.append(a)
                    else:
                        snap_url = upload_frame_snapshot(crop, f"snapshots/persons/{session_id}")
                        det.snapshot_url = snap_url
                        person_id = db_save_person(
                            embedding, age, gender, gender_conf, snap_url,
                            age_range, accessories, height_cm, clothing_color, session_id
                        )
                        det.person_id = person_id
                        det.is_returning = False

                        known_face_embeddings.append({
                            "id": person_id,
                            "name": None,
                            "alias": None,
                            "embedding": embedding,
                            "flagged": False,
                            "flag_reason": None,
                            "threat_level": "none",
                            "threat_score": 0.0,
                            "age_estimate": age,
                            "gender": gender,
                        })

                    if accessories.get("mask"):
                        a = AlertResult(
                            alert_type="masked_person_detected",
                            severity="medium",
                            description="Person with face covering detected",
                            person_id=det.person_id,
                            metadata={"track_id": track_id}
                        )
                        alerts.append(a)

                    demographic_counters[session_id]["gender_" + (gender or "U")] += 1
                    if age > 0:
                        demographic_counters[session_id]["age_" + compute_age_range(age)] += 1
                    if det.emotion:
                        demographic_counters[session_id]["emotion_" + det.emotion] += 1

                if track_id >= 0:
                    loiter_key = f"{session_id}_{track_id}"
                    if loiter_key not in loiter_tracker:
                        loiter_tracker[loiter_key] = time_now
                    else:
                        loiter_duration = time_now - loiter_tracker[loiter_key]
                        if loiter_duration > LOITER_TIME_SECONDS:
                            a = AlertResult(
                                alert_type="loitering_detected",
                                severity="medium",
                                description=f"Person loitering for {int(loiter_duration)}s",
                                person_id=det.person_id,
                                track_id=track_id,
                                metadata={"duration_s": round(loiter_duration, 1)}
                            )
                            alerts.append(a)
                            det.behavior_flags.append("loitering")
                            db_save_behavior_event(session_id, "loitering", det.person_id, track_id,
                                                   loiter_duration, 0.85, {"cx": cx, "cy": cy})
                            loiter_tracker[loiter_key] = time_now

                    pos_hist = track_position_histories[track_id]
                    if len(pos_hist) >= 3:
                        recent_movement = [
                            math.sqrt((pos_hist[i][0]-pos_hist[i-1][0])**2 + (pos_hist[i][1]-pos_hist[i-1][1])**2)
                            for i in range(-3, 0)
                        ]
                        avg_movement = np.mean(recent_movement)
                        if avg_movement > RUNNING_SPEED_THRESHOLD * 10:
                            det.behavior_flags.append("running")
                            if "running" not in [a.alert_type for a in alerts]:
                                alerts.append(AlertResult(
                                    alert_type="running_detected",
                                    severity="low",
                                    description="Person running detected",
                                    person_id=det.person_id,
                                    track_id=track_id,
                                ))

                for zone in (restricted_zones or []):
                    if len(zone) >= 4:
                        zx1, zy1, zx2, zy2 = zone[0], zone[1], zone[2], zone[3]
                        if zx1 < cx < zx2 and zy1 < cy < zy2:
                            snap_url = upload_frame_snapshot(frame, f"snapshots/intrusion/{session_id}")
                            a = AlertResult(
                                alert_type="restricted_zone_intrusion",
                                severity="high",
                                description="Person detected in restricted zone",
                                person_id=det.person_id,
                                track_id=track_id,
                                snapshot_url=snap_url,
                            )
                            alerts.append(a)
                            det.behavior_flags.append("restricted_zone")
                            zone_intrusion_log[session_id].append({
                                "ts": datetime.datetime.utcnow().isoformat(),
                                "track_id": track_id,
                                "person_id": det.person_id,
                                "zone": zone,
                            })

                yolo_class_name = yolo_results.names.get(cls_id, "")
                if any(w in yolo_class_name.lower() for w in SUSPICIOUS_OBJECTS):
                    is_abandoned = detect_abandoned_object(session_id, track_id, yolo_class_name, (cx, cy), time_now)
                    if is_abandoned:
                        a = AlertResult(
                            alert_type="abandoned_object",
                            severity="high",
                            description=f"Abandoned {yolo_class_name} detected",
                            track_id=track_id,
                        )
                        alerts.append(a)
                        det.behavior_flags.append("abandoned_object")

                detections.append(det)

            elif cls_id in VEHICLE_CLASSES and conf > VEHICLE_CONFIDENCE_THRESHOLD:
                vehicle_type = VEHICLE_CLASSES[cls_id]
                vehicle_color, vehicle_color_hex = compute_dominant_color(crop)

                if track_id >= 0:
                    track_frame_counts[track_id] += 1
                    track_position_histories[track_id].append([cx, cy])
                    if len(track_position_histories[track_id]) > HISTORY_LEN:
                        track_position_histories[track_id].pop(0)

                speed_kph = 0.0
                if perspective_matrix is not None and track_frame_counts[track_id] >= MIN_TRACK_FRAMES:
                    speed_kph = compute_vehicle_speed(
                        track_id, track_position_histories[track_id], fps, perspective_matrix
                    )

                det = DetectionResult(
                    det_type="vehicle",
                    bbox=xyxy,
                    confidence=conf,
                    track_id=track_id,
                    vehicle_type=vehicle_type,
                    vehicle_color=vehicle_color,
                    speed_kph=speed_kph if speed_kph > 0 else None,
                    night_mode=night_active,
                )

                if frame_count % PLATE_SCAN_INTERVAL_FRAMES == 0:
                    processed_crop = preprocess_plate_crop(crop)
                    plate_text, plate_conf = run_plate_ocr(processed_crop)
                    if not plate_text and len(crop) > 0:
                        plate_text, plate_conf = run_plate_ocr(crop)

                    if plate_text and plate_conf > PLATE_OCR_CONFIDENCE:
                        det.plate = plate_text
                        det.plate_confidence = plate_conf
                        snap_url = upload_frame_snapshot(crop, f"snapshots/plates/{session_id}")
                        vehicle_id = db_save_vehicle(
                            plate_text, plate_conf, vehicle_type, vehicle_color, vehicle_color_hex, snap_url
                        )
                        det.vehicle_id = vehicle_id

                        if speed_kph > 0:
                            db_update_vehicle_speed(vehicle_id, speed_kph)

                        db_save_plate_read(session_id, plate_text, plate_conf, vehicle_id, speed_kph, snap_url,
                                           plate_text in known_plate_registry and known_plate_registry[plate_text].get("flagged", False))

                        known_plate_registry[plate_text] = {"id": vehicle_id, "plate_number": plate_text,
                                                             "flagged": False, "stolen": False, "seen_count": 1}

                        plate_info = known_plate_registry.get(plate_text, {})
                        if plate_info.get("flagged"):
                            det.is_flagged = True
                            a = AlertResult(
                                alert_type="flagged_vehicle_detected",
                                severity="critical",
                                description=f"Flagged vehicle plate {plate_text} detected",
                                vehicle_id=vehicle_id,
                                plate=plate_text,
                                speed_kph=speed_kph,
                                snapshot_url=snap_url,
                                metadata={"flag_reason": plate_info.get("flag_reason")}
                            )
                            alerts.append(a)
                        if plate_info.get("stolen"):
                            a = AlertResult(
                                alert_type="stolen_vehicle_detected",
                                severity="critical",
                                description=f"Stolen vehicle plate {plate_text} detected",
                                vehicle_id=vehicle_id,
                                plate=plate_text,
                                speed_kph=speed_kph,
                                snapshot_url=snap_url,
                            )
                            alerts.append(a)

                if speed_kph > 120:
                    a = AlertResult(
                        alert_type="speeding_detected",
                        severity="high",
                        description=f"Vehicle exceeding speed limit: {speed_kph} km/h",
                        vehicle_id=det.vehicle_id,
                        plate=det.plate,
                        speed_kph=speed_kph,
                    )
                    alerts.append(a)

                vehicle_tracks_for_tailgate.append({
                    "track_id": track_id, "cx": cx, "cy": cy, "speed": speed_kph
                })
                detections.append(det)

    if frame_count % POSE_ANALYSIS_INTERVAL_FRAMES == 0 and pose_results.keypoints is not None:
        for kp_set in pose_results.keypoints.xy:
            kp_array = kp_set.cpu().numpy()
            pose_analysis = analyze_pose_keypoints(kp_array)
            if pose_analysis["threat"]:
                snap_url = upload_frame_snapshot(frame, f"snapshots/pose_threats/{session_id}")
                a = AlertResult(
                    alert_type="threat_posture_detected",
                    severity="high",
                    description=f"Threatening body posture: {', '.join(pose_analysis['reasons'])}",
                    snapshot_url=snap_url,
                    metadata={"reasons": pose_analysis["reasons"]}
                )
                alerts.append(a)

            gait_feats = pose_analysis.get("gait_features", [])
            if gait_feats:
                gait_signature_store[session_id].append(gait_feats)
                if len(gait_signature_store[session_id]) > GAIT_WINDOW:
                    gait_signature_store[session_id].pop(0)

    fight_detected = detect_fight_motion(session_id, flow_mag)
    if fight_detected and flow_mag > 0:
        a = AlertResult(
            alert_type="fight_detected",
            severity="critical",
            description=f"High-velocity chaotic motion detected (flow: {flow_mag:.1f})",
            metadata={"flow_magnitude": round(flow_mag, 2)}
        )
        alerts.append(a)

    person_count = len(person_positions_for_crowd)
    if person_count >= CROWD_DENSITY_ALERT_THRESHOLD:
        update_crowd_density(session_id, frame.shape, person_positions_for_crowd)
        a = AlertResult(
            alert_type="crowd_density_alert",
            severity="medium" if person_count < 20 else "high",
            description=f"High crowd density: {person_count} persons in frame",
            metadata={"count": person_count}
        )
        alerts.append(a)

    tailgate_alerts = detect_tailgating(session_id, vehicle_tracks_for_tailgate)
    for tg in tailgate_alerts:
        a = AlertResult(
            alert_type="vehicle_tailgating",
            severity="medium",
            description=f"Vehicles too close: tracks {tg['track_a']} and {tg['track_b']}",
            metadata=tg
        )
        alerts.append(a)

    if enable_heatmap:
        update_attention_heatmap(session_id, frame.shape, detections)

    return detections, alerts

def build_hud_overlay(
    frame: np.ndarray,
    detections: List[DetectionResult],
    alerts: List[AlertResult],
    session_id: str,
    frame_count: int,
    fps: float,
    total_alerts: int,
    enable_heatmap: bool,
    night_mode: bool,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if enable_heatmap and session_id in attention_heatmap:
        out = render_heatmap_overlay(out, session_id)

    top_bar_h = 44
    cv2.rectangle(out, (0, 0), (w, top_bar_h), (5, 5, 5), -1)
    cv2.addWeighted(out[:top_bar_h], 0.75, frame[:top_bar_h], 0.25, 0, out[:top_bar_h])

    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")
    status_color = (0, 60, 255) if any(a.severity == "critical" for a in alerts) else (0, 200, 50)
    cv2.putText(out, f"  ARGUS  |  {ts}  |  FPS {fps:.1f}  |  ALERTS {total_alerts}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)

    if night_mode:
        cv2.putText(out, "[NIGHT]", (w - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)

    bottom_bar_y = h - 28
    cv2.rectangle(out, (0, bottom_bar_y), (w, h), (5, 5, 5), -1)
    cv2.putText(out, f"  SESSION {session_id[:12]}  |  FRAME {frame_count}  |  DETECTIONS {len(detections)}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 240), 1, cv2.LINE_AA)

    for det in detections:
        if len(det.bbox) < 4:
            continue
        x1, y1, x2, y2 = int(det.bbox[0]), int(det.bbox[1]), int(det.bbox[2]), int(det.bbox[3])

        if det.is_flagged or det.det_type == "weapon":
            color = (0, 0, 255)
        elif det.det_type == "vehicle":
            color = (0, 140, 255)
        elif det.is_returning:
            color = (255, 60, 220)
        elif det.pose_threat:
            color = (0, 80, 255)
        else:
            color = (30, 230, 30)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        cl = 14
        for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out, (px, py), (px + dx*cl, py), color, 2)
            cv2.line(out, (px, py), (px, py + dy*cl), color, 2)

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(out, (center_x, center_y), color, cv2.MARKER_CROSS, 10, 1)

        label_parts = [det.det_type.upper()]
        if det.name:
            label_parts.append(det.name)
        if det.is_returning:
            label_parts.append("RETURNING")
        if det.is_flagged:
            label_parts.append("⚠ FLAGGED")
        if det.emotion:
            label_parts.append(det.emotion.upper())
        if det.age:
            label_parts.append(f"~{det.age}yr")
        if det.gender:
            label_parts.append(det.gender)
        if det.mask_detected:
            label_parts.append("MASKED")
        if det.glasses_detected:
            label_parts.append("GLASSES")
        if det.speed_kph:
            spd_color = (0, 255, 0) if det.speed_kph < 60 else (0, 165, 255) if det.speed_kph < 100 else (0, 0, 255)
            cv2.putText(out, f"{det.speed_kph}km/h", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, spd_color, 2, cv2.LINE_AA)
        if det.plate:
            cv2.putText(out, f"[{det.plate}]", (x1, y2 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 2, cv2.LINE_AA)
        if det.vehicle_color:
            label_parts.append(det.vehicle_color)
        if det.gait_label and det.gait_label != "normal":
            label_parts.append(f"GAIT:{det.gait_label.upper()}")
        if det.behavior_flags:
            label_parts += [f for f in det.behavior_flags]
        if det.threat_level and det.threat_level != "none":
            label_parts.append(f"TL:{det.threat_level.upper()}")

        label = "  ".join(label_parts)
        label_y = y1 - 10 if y1 > 30 else y2 + 24
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        lx = max(0, min(x1, w - tw - 6))
        cv2.rectangle(out, (lx, label_y - th - 4), (lx + tw + 6, label_y + 3), color, -1)
        cv2.putText(out, label, (lx + 3, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

        if det.pose_threat:
            cv2.putText(out, "THREAT POSTURE", (x1, y1 - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    for i, alert in enumerate(alerts[-5:]):
        ay = top_bar_h + 18 + i * 22
        sev_colors = {"low": (50, 200, 50), "medium": (0, 165, 255), "high": (0, 80, 255), "critical": (0, 0, 255)}
        ac = sev_colors.get(alert.severity, (255, 255, 255))
        cv2.putText(out, f"[{alert.severity.upper()}] {alert.alert_type}: {alert.description[:60]}",
                    (8, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ac, 1, cv2.LINE_AA)

    if session_id in crowd_density_grid:
        grid = crowd_density_grid[session_id]
        for gy in range(grid.shape[0]):
            for gx in range(grid.shape[1]):
                count = int(grid[gy, gx])
                if count > 0:
                    px = gx * HEATMAP_GRID_SIZE
                    py = gy * HEATMAP_GRID_SIZE
                    alpha = min(count / 5.0, 1.0)
                    overlay_color = (0, 0, int(200 * alpha))
                    cv2.rectangle(out, (px, py), (px + HEATMAP_GRID_SIZE, py + HEATMAP_GRID_SIZE),
                                  overlay_color, -1)

    return out

@app.on_event("startup")
async def on_startup():
    try:
        db_connect()
        logger.info("DB pool connected")
        init_database()
        init_redis()
        logger.info("Redis connected")
        init_r2()
        logger.info("R2 connected")
        init_ai_models()
        load_known_faces_from_db()
        load_known_plates_from_db()
        db_log_system("INFO", "startup", "ARGUS system online")
        logger.info("ARGUS fully online")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.on_event("shutdown")
async def on_shutdown():
    if db_pool:
        db_pool.closeall()
    if redis_client:
        await redis_client.close()
    executor.shutdown(wait=False)
    logger.info("ARGUS shutdown complete")

@app.get("/", tags=["system"])
async def root():
    return {
        "system": "ARGUS",
        "version": "2.0.0",
        "status": "online",
        "uptime": "active",
        "capabilities": [
            "face_recognition", "emotion_detection", "age_gender_estimation",
            "gait_analysis", "vehicle_detection", "plate_ocr", "speed_tracking",
            "weapon_detection", "pose_threat_analysis", "crowd_density",
            "behavior_analysis", "audio_threat_detection", "night_mode",
            "attention_heatmap", "loitering_detection", "fight_detection",
            "abandoned_object_detection", "tailgating_detection",
            "optical_flow_analysis", "demographic_analytics", "re_identification",
        ]
    }

@app.get("/health", tags=["system"])
async def health_check():
    db_ok = False
    redis_ok = False
    try:
        db_execute("SELECT 1", fetch="scalar")
        db_ok = True
    except Exception:
        pass
    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:
        pass
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "error",
        "redis": "ok" if redis_ok else "error",
        "known_faces": len(known_face_embeddings),
        "known_plates": len(known_plate_registry),
        "active_sessions": len(active_sessions),
        "active_ws": len(active_ws_connections),
        "flagged_faces": len(flagged_faces_cache),
        "models_loaded": yolo_detector is not None,
    }

@app.post("/auth/token", tags=["auth"])
async def issue_token(x_api_key: str = Header(...)):
    key_data = check_api_key(x_api_key)
    if not key_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    payload = {
        "sub": key_data["name"],
        "permissions": key_data["permissions"],
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return {"token": token, "type": "bearer", "expires_in": 86400}

@app.post("/auth/refresh", tags=["auth"])
async def refresh_token(user=Depends(require_auth)):
    payload = {
        "sub": user.get("sub", "argus_user"),
        "permissions": user.get("permissions", []),
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return {"token": token, "type": "bearer", "expires_in": 86400}

@app.post("/api/keys", tags=["auth"])
async def create_api_key(req: ApiKeyCreateRequest, user=Depends(require_admin)):
    new_key = "argus_" + uuid.uuid4().hex + uuid.uuid4().hex[:8]
    key_hash = hashlib.sha256(new_key.encode()).hexdigest()
    expires_at = None
    if req.expires_in_days:
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=req.expires_in_days)
    db_execute(
        "INSERT INTO api_keys (key_hash, name, permissions, rate_limit_per_hour, expires_at, created_by) VALUES (%s,%s,%s::jsonb,%s,%s,%s)",
        (key_hash, req.name, json.dumps(req.permissions), req.rate_limit_per_hour, expires_at, user.get("sub", "admin"))
    )
    return {"key": new_key, "name": req.name, "permissions": req.permissions, "expires_at": expires_at}

@app.get("/api/keys", tags=["auth"])
async def list_api_keys(user=Depends(require_admin)):
    rows = db_execute("SELECT id, name, created_at, last_used, active, request_count, permissions FROM api_keys ORDER BY created_at DESC", fetch="all")
    return [dict(r) for r in (rows or [])]

@app.delete("/api/keys/{key_id}", tags=["auth"])
async def revoke_api_key(key_id: str, user=Depends(require_admin)):
    db_execute("UPDATE api_keys SET active = FALSE WHERE id = %s", (key_id,))
    return {"status": "revoked", "key_id": key_id}

@app.post("/operators", tags=["auth"])
async def create_operator(req: OperatorCreateRequest, user=Depends(require_admin)):
    import hashlib as hl
    pw_hash = hl.sha256(req.password.encode()).hexdigest()
    db_execute(
        "INSERT INTO operators (username, password_hash, role, permissions) VALUES (%s,%s,%s,%s::jsonb)",
        (req.username, pw_hash, req.role, json.dumps(req.permissions))
    )
    return {"status": "created", "username": req.username}

@app.post("/sessions/start", tags=["sessions"])
async def start_session(req: SessionStartRequest, user=Depends(require_auth)):
    if len(active_sessions) >= MAX_SESSIONS:
        raise HTTPException(status_code=429, detail=f"Max sessions ({MAX_SESSIONS}) reached")
    session_id = uuid.uuid4().hex
    perspective_config = {}
    if req.perspective_src and req.perspective_dst:
        perspective_config = {"src": req.perspective_src, "dst": req.perspective_dst}
    db_save_session(
        session_id, req.source_type, req.source_url or "",
        req.camera_index, req.enable_audio, req.restricted_zones or [],
        perspective_config, req.operator
    )
    active_sessions[session_id] = {
        "id": session_id,
        "source_type": req.source_type,
        "source_url": req.source_url,
        "camera_index": req.camera_index,
        "perspective_src": req.perspective_src,
        "perspective_dst": req.perspective_dst,
        "enable_audio": req.enable_audio,
        "enable_heatmap": req.enable_heatmap,
        "enable_night_mode": req.enable_night_mode,
        "restricted_zones": req.restricted_zones or [],
        "operator": req.operator,
        "running": True,
        "started_at": time.time(),
        "frame_count": 0,
        "processed_frames": 0,
        "detection_count": 0,
        "alert_count": 0,
        "unique_persons": set(),
        "unique_vehicles": set(),
        "fps": 0.0,
    }
    db_log_system("INFO", "sessions", f"Session started: {session_id}", session_id)
    return {"session_id": session_id, "status": "started", "started_at": datetime.datetime.utcnow().isoformat()}

@app.post("/sessions/{session_id}/stop", tags=["sessions"])
async def stop_session(session_id: str, user=Depends(require_auth)):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = active_sessions[session_id]
    s["running"] = False
    duration = time.time() - s["started_at"]
    db_update_session_stats(session_id, {
        **s,
        "unique_persons": len(s.get("unique_persons", set())),
        "unique_vehicles": len(s.get("unique_vehicles", set())),
    })
    db_end_session(session_id, duration)
    del active_sessions[session_id]
    db_log_system("INFO", "sessions", f"Session stopped: {session_id}", session_id)
    return {"session_id": session_id, "status": "stopped", "duration_seconds": round(duration, 2)}

@app.get("/sessions", tags=["sessions"])
async def list_sessions(limit: int = 50, user=Depends(require_auth)):
    rows = db_execute("SELECT * FROM sessions ORDER BY created_at DESC LIMIT %s", (limit,), fetch="all")
    result = [dict(r) for r in (rows or [])]
    for r in result:
        if r["id"] in active_sessions:
            r["active"] = True
    return result

@app.get("/sessions/{session_id}", tags=["sessions"])
async def get_session(session_id: str, user=Depends(require_auth)):
    if session_id in active_sessions:
        s = dict(active_sessions[session_id])
        s["unique_persons"] = len(s.get("unique_persons", set()))
        s["unique_vehicles"] = len(s.get("unique_vehicles", set()))
        s["active"] = True
        return s
    row = db_execute("SELECT * FROM sessions WHERE id = %s", (session_id,), fetch="one")
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return dict(row)

@app.get("/sessions/{session_id}/detections", tags=["sessions"])
async def get_session_detections(session_id: str, limit: int = 200, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT * FROM detections WHERE session_id = %s ORDER BY detected_at DESC LIMIT %s",
        (session_id, limit), fetch="all"
    )
    return [dict(r) for r in (rows or [])]

@app.get("/sessions/{session_id}/alerts", tags=["sessions"])
async def get_session_alerts(session_id: str, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT * FROM alerts WHERE session_id = %s ORDER BY created_at DESC",
        (session_id,), fetch="all"
    )
    return [dict(r) for r in (rows or [])]

@app.get("/sessions/{session_id}/zone-intrusions", tags=["sessions"])
async def get_zone_intrusions(session_id: str, user=Depends(require_auth)):
    return zone_intrusion_log.get(session_id, [])

@app.get("/sessions/{session_id}/demographics", tags=["sessions"])
async def get_session_demographics(session_id: str, user=Depends(require_auth)):
    counters = demographic_counters.get(session_id, Counter())
    gender = {k.replace("gender_", ""): v for k, v in counters.items() if k.startswith("gender_")}
    age = {k.replace("age_", ""): v for k, v in counters.items() if k.startswith("age_")}
    emotion = {k.replace("emotion_", ""): v for k, v in counters.items() if k.startswith("emotion_")}
    return {"session_id": session_id, "gender": gender, "age_ranges": age, "emotions": emotion}

@app.get("/persons", tags=["persons"])
async def list_persons(
    limit: int = 50, offset: int = 0,
    flagged: Optional[bool] = None,
    threat_level: Optional[str] = None,
    gender: Optional[str] = None,
    user=Depends(require_auth)
):
    query = "SELECT id, name, alias, age_estimate, gender, threat_level, threat_score, flagged, flag_reason, first_seen, last_seen, seen_count, snapshot_url, mask_detected, glasses_detected, height_estimate_cm, clothing_color FROM persons WHERE 1=1"
    params = []
    if flagged is not None:
        query += " AND flagged = %s"
        params.append(flagged)
    if threat_level:
        query += " AND threat_level = %s"
        params.append(threat_level)
    if gender:
        query += " AND gender = %s"
        params.append(gender)
    query += " ORDER BY last_seen DESC LIMIT %s OFFSET %s"
    params += [limit, offset]
    rows = db_execute(query, tuple(params), fetch="all")
    return [dict(r) for r in (rows or [])]

@app.get("/persons/{person_id}", tags=["persons"])
async def get_person(person_id: str, user=Depends(require_auth)):
    row = db_execute(
        "SELECT id, name, alias, age_estimate, age_range, gender, threat_level, threat_score, flagged, flag_reason, flag_timestamp, first_seen, last_seen, seen_count, snapshot_url, additional_snapshots, location_log, behavior_tags, dominant_emotion, emotion_history, gait_signature, mask_detected, glasses_detected, height_estimate_cm, clothing_color, notes FROM persons WHERE id = %s",
        (person_id,), fetch="one"
    )
    if not row:
        raise HTTPException(status_code=404, detail="Person not found")
    result = dict(row)
    detections = db_execute(
        "SELECT id, detected_at, session_id, emotion, speed_kph, pose_threat, behavior_flags, night_mode FROM detections WHERE person_id = %s ORDER BY detected_at DESC LIMIT 20",
        (person_id,), fetch="all"
    )
    result["recent_detections"] = [dict(d) for d in (detections or [])]
    alerts_list = db_execute(
        "SELECT id, alert_type, severity, created_at, description FROM alerts WHERE person_id = %s ORDER BY created_at DESC LIMIT 10",
        (person_id,), fetch="all"
    )
    result["alerts"] = [dict(a) for a in (alerts_list or [])]
    watchlist_entries = db_execute(
        "SELECT id, list_name, reason, priority, added_at FROM watchlist WHERE person_id = %s AND active = TRUE",
        (person_id,), fetch="all"
    )
    result["watchlist"] = [dict(w) for w in (watchlist_entries or [])]
    return result

@app.post("/persons/flag", tags=["persons"])
async def flag_person(req: PersonFlagRequest, user=Depends(require_auth)):
    db_execute(
        "UPDATE persons SET flagged = TRUE, flag_reason = %s, threat_level = %s, flag_timestamp = NOW(), flag_operator = %s WHERE id = %s",
        (req.reason, req.threat_level, req.operator, req.person_id)
    )
    load_known_faces_from_db()
    db_log_system("WARN", "persons", f"Person flagged: {req.person_id} — {req.reason}")
    return {"status": "flagged", "person_id": req.person_id}

@app.post("/persons/unflag", tags=["persons"])
async def unflag_person(person_id: str, user=Depends(require_auth)):
    db_execute(
        "UPDATE persons SET flagged = FALSE, flag_reason = NULL, threat_level = 'none' WHERE id = %s",
        (person_id,)
    )
    load_known_faces_from_db()
    return {"status": "unflagged", "person_id": person_id}

@app.post("/persons/name", tags=["persons"])
async def name_person(req: PersonNameRequest, user=Depends(require_auth)):
    db_execute(
        "UPDATE persons SET name = %s, alias = %s, notes = %s WHERE id = %s",
        (req.name, req.alias, req.notes, req.person_id)
    )
    load_known_faces_from_db()
    return {"status": "named", "person_id": req.person_id, "name": req.name}

@app.post("/persons/link", tags=["persons"])
async def link_persons(req: PersonLinkRequest, user=Depends(require_auth)):
    db_execute(
        "UPDATE persons SET known_associates = known_associates || %s::jsonb WHERE id = %s",
        (json.dumps([{"id": req.person_id_b, "relationship": req.relationship}]), req.person_id_a)
    )
    db_execute(
        "UPDATE persons SET known_associates = known_associates || %s::jsonb WHERE id = %s",
        (json.dumps([{"id": req.person_id_a, "relationship": req.relationship}]), req.person_id_b)
    )
    return {"status": "linked", "person_a": req.person_id_a, "person_b": req.person_id_b}

@app.delete("/persons/{person_id}", tags=["persons"])
async def delete_person(person_id: str, user=Depends(require_admin)):
    db_execute("DELETE FROM persons WHERE id = %s", (person_id,))
    load_known_faces_from_db()
    return {"status": "deleted", "person_id": person_id}

@app.post("/persons/{person_id}/enroll", tags=["persons"])
async def enroll_person_face(person_id: str, file: UploadFile = File(...), user=Depends(require_auth)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    faces = detect_faces_on_crop(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in image")
    embedding = faces[0]["embedding"]
    emb_list = embedding.tolist()
    snap_url = upload_frame_snapshot(img, f"snapshots/enrolled/{person_id}")
    db_execute(
        "UPDATE persons SET embedding = %s::vector, snapshot_url = %s WHERE id = %s",
        (json.dumps(emb_list), snap_url, person_id)
    )
    load_known_faces_from_db()
    return {"status": "enrolled", "person_id": person_id, "snapshot_url": snap_url}

@app.post("/persons/search/face", tags=["persons"])
async def search_by_face(file: UploadFile = File(...), user=Depends(require_auth)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    faces = detect_faces_on_crop(img)
    if not faces:
        return {"found": False, "message": "No face detected"}
    embedding = faces[0]["embedding"]
    match = match_face_embedding(embedding)
    if not match:
        return {"found": False, "message": "No match found"}
    row = db_execute(
        "SELECT id, name, alias, age_estimate, gender, threat_level, flagged, flag_reason, first_seen, last_seen, seen_count, snapshot_url FROM persons WHERE id = %s",
        (match["id"],), fetch="one"
    )
    return {
        "found": True,
        "similarity": match.get("similarity_score"),
        "distance": match.get("distance"),
        "person": dict(row) if row else None,
    }

@app.get("/persons/{person_id}/gait", tags=["persons"])
async def get_gait_profile(person_id: str, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT * FROM gait_profiles WHERE person_id = %s ORDER BY captured_at DESC LIMIT 20",
        (person_id,), fetch="all"
    )
    return [dict(r) for r in (rows or [])]

@app.get("/vehicles", tags=["vehicles"])
async def list_vehicles(
    limit: int = 50, offset: int = 0,
    flagged: Optional[bool] = None,
    stolen: Optional[bool] = None,
    user=Depends(require_auth)
):
    query = "SELECT * FROM vehicles WHERE 1=1"
    params = []
    if flagged is not None:
        query += " AND flagged = %s"
        params.append(flagged)
    if stolen is not None:
        query += " AND stolen = %s"
        params.append(stolen)
    query += " ORDER BY last_seen DESC LIMIT %s OFFSET %s"
    params += [limit, offset]
    rows = db_execute(query, tuple(params), fetch="all")
    return [dict(r) for r in (rows or [])]

@app.get("/vehicles/{vehicle_id}", tags=["vehicles"])
async def get_vehicle(vehicle_id: str, user=Depends(require_auth)):
    row = db_execute("SELECT * FROM vehicles WHERE id = %s", (vehicle_id,), fetch="one")
    if not row:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    result = dict(row)
    plate_reads = db_execute(
        "SELECT * FROM plate_reads WHERE vehicle_id = %s ORDER BY read_at DESC LIMIT 20",
        (vehicle_id,), fetch="all"
    )
    result["plate_reads"] = [dict(r) for r in (plate_reads or [])]
    return result

@app.post("/vehicles/flag", tags=["vehicles"])
async def flag_vehicle(req: VehicleFlagRequest, user=Depends(require_auth)):
    db_execute(
        "UPDATE vehicles SET flagged = TRUE, flag_reason = %s, stolen = %s, flag_timestamp = NOW() WHERE id = %s",
        (req.reason, req.stolen, req.vehicle_id)
    )
    load_known_plates_from_db()
    return {"status": "flagged", "vehicle_id": req.vehicle_id}

@app.post("/vehicles/unflag", tags=["vehicles"])
async def unflag_vehicle(vehicle_id: str, user=Depends(require_auth)):
    db_execute("UPDATE vehicles SET flagged = FALSE, flag_reason = NULL, stolen = FALSE WHERE id = %s", (vehicle_id,))
    load_known_plates_from_db()
    return {"status": "unflagged", "vehicle_id": vehicle_id}

@app.get("/vehicles/search/plate", tags=["vehicles"])
async def search_by_plate(q: str = Query(..., min_length=2), user=Depends(require_auth)):
    rows = db_execute(
        "SELECT * FROM vehicles WHERE plate_number ILIKE %s ORDER BY last_seen DESC LIMIT 20",
        (f"%{q}%",), fetch="all"
    )
    return [dict(r) for r in (rows or [])]

@app.get("/vehicles/{vehicle_id}/history", tags=["vehicles"])
async def get_vehicle_history(vehicle_id: str, user=Depends(require_auth)):
    plate_reads = db_execute(
        "SELECT * FROM plate_reads WHERE vehicle_id = %s ORDER BY read_at DESC",
        (vehicle_id,), fetch="all"
    )
    detections = db_execute(
        "SELECT id, detected_at, session_id, speed_kph, frame_snapshot_url FROM detections WHERE vehicle_id = %s ORDER BY detected_at DESC LIMIT 50",
        (vehicle_id,), fetch="all"
    )
    return {
        "vehicle_id": vehicle_id,
        "plate_reads": [dict(r) for r in (plate_reads or [])],
        "detections": [dict(d) for d in (detections or [])],
    }

@app.get("/alerts", tags=["alerts"])
async def list_alerts(
    limit: int = 100, offset: int = 0,
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    resolved: bool = False,
    session_id: Optional[str] = None,
    user=Depends(require_auth)
):
    query = "SELECT * FROM alerts WHERE resolved = %s"
    params = [resolved]
    if severity:
        query += " AND severity = %s"
        params.append(severity)
    if alert_type:
        query += " AND alert_type ILIKE %s"
        params.append(f"%{alert_type}%")
    if session_id:
        query += " AND session_id = %s"
        params.append(session_id)
    query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params += [limit, offset]
    rows = db_execute(query, tuple(params), fetch="all")
    return [dict(r) for r in (rows or [])]

@app.get("/alerts/critical/live", tags=["alerts"])
async def get_live_critical_alerts(user=Depends(require_auth)):
    rows = db_execute(
        "SELECT * FROM alerts WHERE severity = 'critical' AND resolved = FALSE ORDER BY created_at DESC LIMIT 50",
        fetch="all"
    )
    return [dict(r) for r in (rows or [])]

@app.post("/alerts/resolve", tags=["alerts"])
async def resolve_alert(req: AlertResolveRequest, user=Depends(require_auth)):
    db_execute(
        "UPDATE alerts SET resolved = TRUE, resolved_at = NOW(), resolved_by = %s, resolution_note = %s WHERE id = %s",
        (req.resolved_by, req.resolution_note, req.alert_id)
    )
    return {"status": "resolved", "alert_id": req.alert_id}

@app.post("/alerts/acknowledge", tags=["alerts"])
async def acknowledge_alert(req: AlertAcknowledgeRequest, user=Depends(require_auth)):
    db_execute(
        "UPDATE alerts SET acknowledged = TRUE, acknowledged_at = NOW(), acknowledged_by = %s WHERE id = %s",
        (req.acknowledged_by, req.alert_id)
    )
    return {"status": "acknowledged", "alert_id": req.alert_id}

@app.get("/alerts/{alert_id}", tags=["alerts"])
async def get_alert(alert_id: str, user=Depends(require_auth)):
    row = db_execute("SELECT * FROM alerts WHERE id = %s", (alert_id,), fetch="one")
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    return dict(row)

@app.get("/detections", tags=["detections"])
async def list_detections(
    session_id: Optional[str] = None,
    person_id: Optional[str] = None,
    det_type: Optional[str] = None,
    limit: int = 100, offset: int = 0,
    user=Depends(require_auth)
):
    query = "SELECT * FROM detections WHERE 1=1"
    params = []
    if session_id:
        query += " AND session_id = %s"
        params.append(session_id)
    if person_id:
        query += " AND person_id = %s"
        params.append(person_id)
    if det_type:
        query += " AND detection_type = %s"
        params.append(det_type)
    query += " ORDER BY detected_at DESC LIMIT %s OFFSET %s"
    params += [limit, offset]
    rows = db_execute(query, tuple(params), fetch="all")
    return [dict(r) for r in (rows or [])]

@app.get("/stats/global", tags=["analytics"])
async def global_stats(user=Depends(require_auth)):
    total_persons = db_execute("SELECT COUNT(*) FROM persons", fetch="scalar")
    flagged_persons = db_execute("SELECT COUNT(*) FROM persons WHERE flagged = TRUE", fetch="scalar")
    total_vehicles = db_execute("SELECT COUNT(*) FROM vehicles", fetch="scalar")
    stolen_vehicles = db_execute("SELECT COUNT(*) FROM vehicles WHERE stolen = TRUE", fetch="scalar")
    open_alerts = db_execute("SELECT COUNT(*) FROM alerts WHERE resolved = FALSE", fetch="scalar")
    critical_open = db_execute("SELECT COUNT(*) FROM alerts WHERE resolved = FALSE AND severity = 'critical'", fetch="scalar")
    detections_24h = db_execute("SELECT COUNT(*) FROM detections WHERE detected_at > NOW() - INTERVAL '24 hours'", fetch="scalar")
    detections_1h = db_execute("SELECT COUNT(*) FROM detections WHERE detected_at > NOW() - INTERVAL '1 hour'", fetch="scalar")
    alerts_24h = db_execute("SELECT COUNT(*) FROM alerts WHERE created_at > NOW() - INTERVAL '24 hours'", fetch="scalar")
    total_sessions = db_execute("SELECT COUNT(*) FROM sessions", fetch="scalar")
    audio_events_24h = db_execute("SELECT COUNT(*) FROM audio_events WHERE detected_at > NOW() - INTERVAL '24 hours'", fetch="scalar")
    behavior_events_24h = db_execute("SELECT COUNT(*) FROM behavior_events WHERE detected_at > NOW() - INTERVAL '24 hours'", fetch="scalar")
    return {
        "persons": {"total": total_persons, "flagged": flagged_persons},
        "vehicles": {"total": total_vehicles, "stolen": stolen_vehicles},
        "alerts": {"open": open_alerts, "critical_open": critical_open, "last_24h": alerts_24h},
        "detections": {"last_24h": detections_24h, "last_1h": detections_1h},
        "sessions": {"total": total_sessions, "active": len(active_sessions)},
        "audio_events_24h": audio_events_24h,
        "behavior_events_24h": behavior_events_24h,
        "faces_in_memory": len(known_face_embeddings),
        "flagged_faces_cache": len(flagged_faces_cache),
        "plates_in_registry": len(known_plate_registry),
        "active_ws_connections": len(active_ws_connections),
    }

@app.get("/stats/detections/by-type", tags=["analytics"])
async def detections_by_type(days: int = 7, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT detection_type, COUNT(*) as count FROM detections WHERE detected_at > NOW() - INTERVAL '%s days' GROUP BY detection_type ORDER BY count DESC",
        (days,), fetch="all"
    )
    return [dict(r) for r in (rows or [])]

@app.get("/stats/alerts/by-severity", tags=["analytics"])
async def alerts_by_severity(days: int = 7, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT severity, COUNT(*) as count FROM alerts WHERE created_at > NOW() - INTERVAL '%s days' GROUP BY severity ORDER BY count DESC",
        (days,), fetch="all"
    )
    return [dict(r) for r in (rows or [])]

@app.get("/stats/persons/top-seen", tags=["analytics"])
async def top_seen_persons(limit: int = 10, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, name, seen_count, last_seen, threat_level, flagged FROM persons ORDER BY seen_count DESC LIMIT %s",
        (limit,), fetch="all"
    )
    return [dict(r) for r in (rows or [])]

@app.get("/stats/speed/violations", tags=["analytics"])
async def speed_violations(threshold: float = 100.0, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT plate_number, speed_at_read, read_at, session_id FROM plate_reads WHERE speed_at_read > %s ORDER BY read_at DESC LIMIT 100",
        (threshold,), fetch="all"
    )
    return [dict(r) for r in (rows or [])]

@app.get("/stats/audio-events", tags=["analytics"])
async def audio_event_stats(days: int = 7, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT event_type, COUNT(*) as count, AVG(confidence) as avg_confidence FROM audio_events WHERE detected_at > NOW() - INTERVAL '%s days' GROUP BY event_type ORDER BY count DESC",
        (days,), fetch="all"
    )
    return [dict(r) for r in (rows or [])]

@app.get("/stats/behavior", tags=["analytics"])
async def behavior_stats(session_id: Optional[str] = None, user=Depends(require_auth)):
    if session_id:
        rows = db_execute(
            "SELECT behavior_type, COUNT(*) as count FROM behavior_events WHERE session_id = %s GROUP BY behavior_type ORDER BY count DESC",
            (session_id,), fetch="all"
        )
    else:
        rows = db_execute(
            "SELECT behavior_type, COUNT(*) as count FROM behavior_events GROUP BY behavior_type ORDER BY count DESC",
            fetch="all"
        )
    return [dict(r) for r in (rows or [])]

@app.post("/analyze/image", tags=["analysis"])
async def analyze_single_image(
    file: UploadFile = File(...),
    enable_heatmap: bool = False,
    enable_night_mode: bool = False,
    user=Depends(require_auth)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    session_id = "img_" + uuid.uuid4().hex[:10]
    loop = asyncio.get_event_loop()
    detections, alerts = await loop.run_in_executor(
        executor,
        process_full_frame,
        frame.copy(), session_id, 1, 30.0, None, [], enable_heatmap, enable_night_mode
    )
    annotated = build_hud_overlay(
        frame.copy(), detections, alerts, session_id, 1, 30.0,
        len(alerts), enable_heatmap, enable_night_mode
    )
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])
    annotated_b64 = base64.b64encode(buf.tobytes()).decode()
    return {
        "session_id": session_id,
        "detections": [asdict(d) for d in detections],
        "alerts": [asdict(a) for a in alerts],
        "annotated_frame_b64": annotated_b64,
        "summary": {
            "persons": sum(1 for d in detections if d.det_type == "person"),
            "vehicles": sum(1 for d in detections if d.det_type == "vehicle"),
            "flagged": sum(1 for d in detections if d.is_flagged),
            "threats": sum(1 for d in detections if d.pose_threat or d.threat_level not in ["none", None]),
            "critical_alerts": sum(1 for a in alerts if a.severity == "critical"),
        }
    }

@app.post("/analyze/face", tags=["analysis"])
async def analyze_face_only(file: UploadFile = File(...), user=Depends(require_auth)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    faces = detect_faces_on_crop(img)
    if not faces:
        return {"faces": [], "count": 0}
    results = []
    for fd in faces:
        emotion = analyze_emotion_on_crop(img)
        accessories = detect_accessories(img)
        match = match_face_embedding(fd["embedding"])
        results.append({
            "bbox": fd["bbox"],
            "age": fd["age"],
            "gender": fd["gender"],
            "det_score": fd["det_score"],
            "emotion": emotion,
            "accessories": accessories,
            "match": {
                "found": match is not None,
                "person_id": match["id"] if match else None,
                "name": match.get("name") if match else None,
                "similarity": match.get("similarity_score") if match else None,
                "flagged": match.get("flagged") if match else None,
                "threat_level": match.get("threat_level") if match else None,
            }
        })
    return {"faces": results, "count": len(results)}

@app.post("/analyze/plate", tags=["analysis"])
async def analyze_plate_only(file: UploadFile = File(...), user=Depends(require_auth)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    processed = preprocess_plate_crop(img)
    plate, confidence = run_plate_ocr(processed)
    if not plate:
        plate, confidence = run_plate_ocr(img)
    registry_info = known_plate_registry.get(plate, {}) if plate else {}
    return {
        "plate": plate,
        "confidence": confidence,
        "found_in_registry": bool(plate and plate in known_plate_registry),
        "flagged": registry_info.get("flagged", False),
        "stolen": registry_info.get("stolen", False),
        "vehicle_id": str(registry_info.get("id", "")) if registry_info.get("id") else None,
    }

@app.get("/watchlist", tags=["watchlist"])
async def list_watchlist(list_name: Optional[str] = None, user=Depends(require_auth)):
    if list_name:
        rows = db_execute(
            "SELECT * FROM watchlist WHERE list_name = %s AND active = TRUE ORDER BY priority DESC",
            (list_name,), fetch="all"
        )
    else:
        rows = db_execute("SELECT * FROM watchlist WHERE active = TRUE ORDER BY priority DESC", fetch="all")
    return [dict(r) for r in (rows or [])]

@app.post("/watchlist/add", tags=["watchlist"])
async def add_to_watchlist(req: WatchlistAddRequest, user=Depends(require_auth)):
    entry_id = str(uuid.uuid4())
    db_execute(
        "INSERT INTO watchlist (id, list_name, list_type, person_id, vehicle_plate, reason, priority, added_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (entry_id, req.list_name, req.list_type, req.person_id, req.vehicle_plate, req.reason, req.priority, req.added_by)
    )
    return {"status": "added", "entry_id": entry_id}

@app.delete("/watchlist/{entry_id}", tags=["watchlist"])
async def remove_from_watchlist(entry_id: str, user=Depends(require_auth)):
    db_execute("UPDATE watchlist SET active = FALSE WHERE id = %s", (entry_id,))
    return {"status": "removed", "entry_id": entry_id}

@app.get("/zones", tags=["zones"])
async def list_zones(session_id: Optional[str] = None, user=Depends(require_auth)):
    if session_id:
        rows = db_execute(
            "SELECT * FROM restricted_zones WHERE session_id = %s AND active = TRUE",
            (session_id,), fetch="all"
        )
    else:
        rows = db_execute("SELECT * FROM restricted_zones WHERE active = TRUE", fetch="all")
    return [dict(r) for r in (rows or [])]

@app.post("/zones", tags=["zones"])
async def create_zone(req: ZoneCreateRequest, user=Depends(require_auth)):
    zone_id = str(uuid.uuid4())
    db_execute(
        "INSERT INTO restricted_zones (id, zone_name, session_id, coordinates, zone_type, alert_severity) VALUES (%s,%s,%s,%s::jsonb,%s,%s)",
        (zone_id, req.zone_name, req.session_id, json.dumps(req.coordinates), req.zone_type, req.alert_severity)
    )
    return {"status": "created", "zone_id": zone_id}

@app.delete("/zones/{zone_id}", tags=["zones"])
async def delete_zone(zone_id: str, user=Depends(require_auth)):
    db_execute("UPDATE restricted_zones SET active = FALSE WHERE id = %s", (zone_id,))
    return {"status": "deleted", "zone_id": zone_id}

@app.get("/plate-reads", tags=["vehicles"])
async def list_plate_reads(
    session_id: Optional[str] = None,
    plate: Optional[str] = None,
    flagged: Optional[bool] = None,
    limit: int = 100,
    user=Depends(require_auth)
):
    query = "SELECT * FROM plate_reads WHERE 1=1"
    params = []
    if session_id:
        query += " AND session_id = %s"
        params.append(session_id)
    if plate:
        query += " AND plate_number ILIKE %s"
        params.append(f"%{plate}%")
    if flagged is not None:
        query += " AND flagged = %s"
        params.append(flagged)
    query += " ORDER BY read_at DESC LIMIT %s"
    params.append(limit)
    rows = db_execute(query, tuple(params), fetch="all")
    return [dict(r) for r in (rows or [])]

@app.get("/audio/events", tags=["audio"])
async def list_audio_events(session_id: Optional[str] = None, limit: int = 100, user=Depends(require_auth)):
    if session_id:
        rows = db_execute(
            "SELECT * FROM audio_events WHERE session_id = %s ORDER BY detected_at DESC LIMIT %s",
            (session_id, limit), fetch="all"
        )
    else:
        rows = db_execute("SELECT * FROM audio_events ORDER BY detected_at DESC LIMIT %s", (limit,), fetch="all")
    return [dict(r) for r in (rows or [])]

@app.get("/audio/live-buffer", tags=["audio"])
async def get_audio_live_buffer(user=Depends(require_auth)):
    return {"events": list(audio_event_buffer), "count": len(audio_event_buffer)}

@app.get("/behavior/events", tags=["analytics"])
async def list_behavior_events(
    session_id: Optional[str] = None,
    behavior_type: Optional[str] = None,
    limit: int = 100,
    user=Depends(require_auth)
):
    query = "SELECT * FROM behavior_events WHERE 1=1"
    params = []
    if session_id:
        query += " AND session_id = %s"
        params.append(session_id)
    if behavior_type:
        query += " AND behavior_type = %s"
        params.append(behavior_type)
    query += " ORDER BY detected_at DESC LIMIT %s"
    params.append(limit)
    rows = db_execute(query, tuple(params), fetch="all")
    return [dict(r) for r in (rows or [])]

@app.get("/export/persons/csv", tags=["export"])
async def export_persons_csv(user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, name, alias, age_estimate, gender, threat_level, flagged, flag_reason, first_seen, last_seen, seen_count, height_estimate_cm, clothing_color, mask_detected, glasses_detected FROM persons ORDER BY last_seen DESC",
        fetch="all"
    )
    if not rows:
        return Response(content="", media_type="text/csv")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows([dict(r) for r in rows])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.read().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=argus_persons.csv"}
    )

@app.get("/export/vehicles/csv", tags=["export"])
async def export_vehicles_csv(user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, plate_number, vehicle_type, vehicle_color, flagged, stolen, first_seen, last_seen, seen_count, avg_speed_kph, max_speed_kph FROM vehicles ORDER BY last_seen DESC",
        fetch="all"
    )
    if not rows:
        return Response(content="", media_type="text/csv")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows([dict(r) for r in rows])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.read().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=argus_vehicles.csv"}
    )

@app.get("/export/alerts/csv", tags=["export"])
async def export_alerts_csv(user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, alert_type, severity, created_at, resolved, description, session_id FROM alerts ORDER BY created_at DESC",
        fetch="all"
    )
    if not rows:
        return Response(content="", media_type="text/csv")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows([dict(r) for r in rows])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.read().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=argus_alerts.csv"}
    )

@app.get("/export/plate-reads/csv", tags=["export"])
async def export_plate_reads_csv(user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, plate_number, confidence, speed_at_read, read_at, session_id, flagged FROM plate_reads ORDER BY read_at DESC",
        fetch="all"
    )
    if not rows:
        return Response(content="", media_type="text/csv")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows([dict(r) for r in rows])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.read().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=argus_plate_reads.csv"}
    )

@app.post("/admin/reload/faces", tags=["admin"])
async def admin_reload_faces(user=Depends(require_admin)):
    load_known_faces_from_db()
    return {"status": "reloaded", "count": len(known_face_embeddings)}

@app.post("/admin/reload/plates", tags=["admin"])
async def admin_reload_plates(user=Depends(require_admin)):
    load_known_plates_from_db()
    return {"status": "reloaded", "count": len(known_plate_registry)}

@app.delete("/admin/sessions/cleanup", tags=["admin"])
async def admin_cleanup_sessions(user=Depends(require_admin)):
    stale = [sid for sid, s in active_sessions.items() if not s.get("running")]
    for sid in stale:
        del active_sessions[sid]
    return {"status": "cleaned", "removed": len(stale)}

@app.get("/admin/system/logs", tags=["admin"])
async def get_system_logs(limit: int = 100, level: Optional[str] = None, user=Depends(require_admin)):
    if level:
        rows = db_execute(
            "SELECT * FROM system_logs WHERE level = %s ORDER BY logged_at DESC LIMIT %s",
            (level, limit), fetch="all"
        )
    else:
        rows = db_execute("SELECT * FROM system_logs ORDER BY logged_at DESC LIMIT %s", (limit,), fetch="all")
    return [dict(r) for r in (rows or [])]

@app.delete("/admin/clear/heatmaps", tags=["admin"])
async def clear_heatmaps(session_id: Optional[str] = None, user=Depends(require_admin)):
    if session_id:
        attention_heatmap.pop(session_id, None)
        crowd_density_grid.pop(session_id, None)
    else:
        attention_heatmap.clear()
        crowd_density_grid.clear()
    return {"status": "cleared"}

@app.get("/live/sessions", tags=["live"])
async def live_sessions(user=Depends(require_auth)):
    result = []
    for sid, s in active_sessions.items():
        entry = dict(s)
        entry["unique_persons"] = len(entry.get("unique_persons", set()))
        entry["unique_vehicles"] = len(entry.get("unique_vehicles", set()))
        entry["duration_s"] = round(time.time() - s.get("started_at", time.time()), 1)
        entry["ws_connected"] = sid in active_ws_connections
        result.append(entry)
    return result

@app.get("/live/connections", tags=["live"])
async def live_connections(user=Depends(require_auth)):
    return {
        "video_streams": list(active_ws_connections.keys()),
        "audio_streams": list(audio_ws_connections.keys()),
        "total": len(active_ws_connections) + len(audio_ws_connections),
    }

@app.get("/live/audio-buffer", tags=["live"])
async def live_audio_buffer(user=Depends(require_auth)):
    return {"buffer": list(audio_event_buffer), "count": len(audio_event_buffer)}

@app.get("/live/heatmap/{session_id}", tags=["live"])
async def get_live_heatmap(session_id: str, user=Depends(require_auth)):
    if session_id not in attention_heatmap:
        return {"session_id": session_id, "heatmap": None}
    hm = attention_heatmap[session_id]
    return {
        "session_id": session_id,
        "grid": hm.tolist(),
        "max": float(np.max(hm)),
        "shape": list(hm.shape),
    }

@app.get("/live/crowd-density/{session_id}", tags=["live"])
async def get_live_crowd_density(session_id: str, user=Depends(require_auth)):
    if session_id not in crowd_density_grid:
        return {"session_id": session_id, "grid": None, "total": 0}
    grid = crowd_density_grid[session_id]
    return {
        "session_id": session_id,
        "grid": grid.tolist(),
        "total_persons": int(np.sum(grid)),
        "hotspot": {"x": int(np.argmax(np.sum(grid, axis=0))), "y": int(np.argmax(np.sum(grid, axis=1)))},
    }

@app.websocket("/ws/stream/{session_id}")
async def websocket_video_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in active_sessions:
        await websocket.send_json({"error": "Session not found", "session_id": session_id})
        await websocket.close(code=1008)
        return

    active_ws_connections[session_id] = websocket
    session = active_sessions[session_id]
    source_type = session["source_type"]
    camera_index = session.get("camera_index", 0)
    source_url = session.get("source_url")
    restricted_zones = session.get("restricted_zones", [])
    enable_heatmap = session.get("enable_heatmap", False)
    enable_night_mode = session.get("enable_night_mode", False)

    perspective_matrix = None
    p_src = session.get("perspective_src")
    p_dst = session.get("perspective_dst")
    if p_src and p_dst:
        try:
            perspective_matrix = cv2.getPerspectiveTransform(
                np.float32(p_src), np.float32(p_dst)
            )
        except Exception as e:
            logger.warning(f"Perspective transform failed: {e}")

    if source_type == "webcam":
        cap = cv2.VideoCapture(camera_index)
    elif source_type in ("rtsp", "ip_cam", "file"):
        cap = cv2.VideoCapture(source_url)
    else:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = native_fps if native_fps and native_fps > 0 else 30.0

    frame_count = 0
    processed_count = 0
    total_alert_count = 0
    total_detection_count = 0
    unique_persons: set = session.setdefault("unique_persons", set())
    unique_vehicles: set = session.setdefault("unique_vehicles", set())
    fps_buffer = deque(maxlen=30)
    prev_frame_time = time.time()
    session_start = time.time()
    last_stat_push = time.time()

    try:
        while session.get("running", False):
            ret, frame = cap.read()
            if not ret:
                if source_type == "file":
                    await websocket.send_json({"event": "stream_ended", "session_id": session_id})
                    break
                await asyncio.sleep(0.03)
                continue

            frame_count += 1
            now = time.time()
            elapsed = now - prev_frame_time
            live_fps = 1.0 / elapsed if elapsed > 0 else fps
            fps_buffer.append(live_fps)
            smooth_fps = float(np.mean(fps_buffer))
            prev_frame_time = now

            if frame_count % FRAME_SKIP != 0 and frame_count > 1:
                continue

            processed_count += 1
            loop = asyncio.get_event_loop()
            detections, alerts = await loop.run_in_executor(
                executor,
                process_full_frame,
                frame.copy(), session_id, frame_count, smooth_fps,
                perspective_matrix, restricted_zones, enable_heatmap, enable_night_mode
            )

            for det in detections:
                total_detection_count += 1
                if det.person_id:
                    unique_persons.add(det.person_id)
                if det.vehicle_id:
                    unique_vehicles.add(det.vehicle_id)

            for alert in alerts:
                total_alert_count += 1
                det_snap = None
                if alert.snapshot_url:
                    det_snap = alert.snapshot_url
                det_id = None
                if detections:
                    matching = [d for d in detections if d.person_id == alert.person_id or d.vehicle_id == alert.vehicle_id]
                    if matching:
                        snap_url = upload_frame_snapshot(frame, f"snapshots/alerts/{session_id}")
                        det_id = db_save_detection(session_id, matching[0], frame_count, snap_url)
                db_save_alert(alert, session_id, det_id)

            active_sessions[session_id].update({
                "frame_count": frame_count,
                "processed_frames": processed_count,
                "detection_count": total_detection_count,
                "alert_count": total_alert_count,
                "fps": round(smooth_fps, 1),
            })

            night_active = night_mode_active.get(session_id, False)
            annotated = build_hud_overlay(
                frame.copy(), detections, alerts, session_id,
                frame_count, smooth_fps, total_alert_count,
                enable_heatmap, night_active
            )

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
            frame_b64 = base64.b64encode(buf.tobytes()).decode()

            payload = {
                "event": "frame",
                "session_id": session_id,
                "frame_count": frame_count,
                "processed": processed_count,
                "fps": round(smooth_fps, 1),
                "frame_b64": frame_b64,
                "night_mode": night_active,
                "detections": [asdict(d) for d in detections],
                "alerts": [asdict(a) for a in alerts],
                "stats": {
                    "unique_persons": len(unique_persons),
                    "unique_vehicles": len(unique_vehicles),
                    "total_alerts": total_alert_count,
                    "total_detections": total_detection_count,
                    "session_duration_s": round(now - session_start, 1),
                },
            }

            await websocket.send_json(payload)

            if now - last_stat_push > 5.0:
                await redis_client.set(
                    f"argus:session:{session_id}:stats",
                    json.dumps({
                        "fps": round(smooth_fps, 1),
                        "alerts": total_alert_count,
                        "detections": total_detection_count,
                        "persons": len(unique_persons),
                        "vehicles": len(unique_vehicles),
                        "updated": now,
                    }),
                    ex=30
                )
                last_stat_push = now

            try:
                raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.005)
                cmd = json.loads(raw_msg) if raw_msg.startswith("{") else {"cmd": raw_msg}
                if cmd.get("cmd") == "stop":
                    break
                if cmd.get("cmd") == "night_toggle":
                    enable_night_mode = not enable_night_mode
                    session["enable_night_mode"] = enable_night_mode
                if cmd.get("cmd") == "heatmap_toggle":
                    enable_heatmap = not enable_heatmap
                    session["enable_heatmap"] = enable_heatmap
            except (asyncio.TimeoutError, Exception):
                pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Stream error [{session_id}]: {e}")
        try:
            await websocket.send_json({"event": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        cap.release()
        active_ws_connections.pop(session_id, None)
        if session_id in active_sessions:
            s = active_sessions[session_id]
            duration = time.time() - session_start
            db_update_session_stats(session_id, {
                **s,
                "unique_persons": len(unique_persons),
                "unique_vehicles": len(unique_vehicles),
            })
            db_end_session(session_id, duration)
            active_sessions.pop(session_id, None)
        optical_flow_prev.pop(session_id, None)
        motion_background_subtractors.pop(session_id, None)
        fight_motion_buffer.pop(session_id, None)
        gait_signature_store.pop(session_id, None)
        logger.info(f"Session cleaned up: {session_id}")

@app.websocket("/ws/audio/{session_id}")
async def websocket_audio_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    audio_ws_connections[session_id] = websocket
    buffer = np.array([], dtype=np.int16)
    try:
        while True:
            raw = await websocket.receive_bytes()
            chunk = np.frombuffer(raw, dtype=np.int16)
            buffer = np.concatenate([buffer, chunk])
            if len(buffer) >= AUDIO_CHUNK_SIZE:
                segment = buffer[:AUDIO_CHUNK_SIZE]
                buffer = buffer[AUDIO_CHUNK_SIZE - AUDIO_OVERLAP:]
                result = classify_audio_chunk(segment, AUDIO_SAMPLE_RATE)
                if result["threat"] and result["type"]:
                    event_id = db_save_audio_event(
                        session_id, result["type"], result["confidence"],
                        result.get("energy", 0.0), result.get("centroid", 0.0)
                    )
                    alert = AlertResult(
                        alert_type=f"audio_threat_{result['type']}",
                        severity="critical" if result["type"] in ["gunshot", "explosion"] else "high",
                        description=f"Audio threat detected: {result['type']} (confidence: {result['confidence']:.2f})",
                        metadata={"event_id": event_id, **result}
                    )
                    db_save_alert(alert, session_id, None)
                    audio_event_buffer.append({
                        "session_id": session_id,
                        "type": result["type"],
                        "confidence": result["confidence"],
                        "ts": datetime.datetime.utcnow().isoformat(),
                    })
                await websocket.send_json({
                    "threat": result["threat"],
                    "type": result.get("type"),
                    "confidence": result.get("confidence", 0.0),
                    "energy": result.get("energy", 0.0),
                    "centroid": result.get("centroid", 0.0),
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Audio WS error: {e}")
    finally:
        audio_ws_connections.pop(session_id, None)

@app.websocket("/ws/alerts/live")
async def websocket_live_alerts(websocket: WebSocket):
    await websocket.accept()
    try:
        last_check = datetime.datetime.utcnow()
        while True:
            await asyncio.sleep(2)
            rows = db_execute(
                "SELECT * FROM alerts WHERE created_at > %s ORDER BY created_at DESC",
                (last_check,), fetch="all"
            )
            if rows:
                last_check = datetime.datetime.utcnow()
                for row in rows:
                    r = dict(row)
                    for k, v in r.items():
                        if isinstance(v, datetime.datetime):
                            r[k] = v.isoformat()
                    await websocket.send_json({"event": "new_alert", "alert": r})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Live alerts WS error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        workers=1,
        ws_ping_interval=20,
        ws_ping_timeout=30,
        timeout_keep_alive=30,
    )


def compute_trajectory_prediction(track_id: int, steps_ahead: int = 10) -> List[Tuple[float, float]]:
    positions = track_position_histories.get(track_id, [])
    if len(positions) < 4:
        return []
    try:
        xs = [p[0] for p in positions[-10:]]
        ys = [p[1] for p in positions[-10:]]
        n = len(xs)
        t = list(range(n))
        vx = (xs[-1] - xs[0]) / max(n - 1, 1)
        vy = (ys[-1] - ys[0]) / max(n - 1, 1)
        ax = 0.0
        ay = 0.0
        if n >= 3:
            vx2 = (xs[-1] - xs[-2])
            vx1 = (xs[-2] - xs[-3])
            ax = vx2 - vx1
            vy2 = (ys[-1] - ys[-2])
            vy1 = (ys[-2] - ys[-3])
            ay = vy2 - vy1
        future = []
        cx, cy = xs[-1], ys[-1]
        for i in range(1, steps_ahead + 1):
            fx = cx + vx * i + 0.5 * ax * i * i
            fy = cy + vy * i + 0.5 * ay * i * i
            future.append((round(fx, 2), round(fy, 2)))
        return future
    except Exception:
        return []


def compute_person_risk_score(det: "DetectionResult", session_id: str) -> float:
    score = 0.0
    if det.is_flagged:
        score += 40.0
    if det.pose_threat:
        score += 25.0
    if det.mask_detected:
        score += 10.0
    if det.emotion in ["angry", "fear", "disgust"]:
        score += 8.0
    if "loitering" in det.behavior_flags:
        score += 10.0
    if "running" in det.behavior_flags:
        score += 6.0
    if "restricted_zone" in det.behavior_flags:
        score += 20.0
    if "abandoned_object" in det.behavior_flags:
        score += 15.0
    threat_map = {"none": 0, "low": 5, "medium": 15, "high": 25, "critical": 40}
    score += threat_map.get(det.threat_level or "none", 0)
    if det.gait_label == "aggressive":
        score += 12.0
    elif det.gait_label == "suspicious":
        score += 8.0
    elif det.gait_label == "running":
        score += 6.0
    return min(round(score, 2), 100.0)


def compute_vehicle_risk_score(det: "DetectionResult") -> float:
    score = 0.0
    if det.is_flagged:
        score += 45.0
    if det.speed_kph and det.speed_kph > 120:
        score += 20.0
    elif det.speed_kph and det.speed_kph > 80:
        score += 8.0
    plate_info = known_plate_registry.get(det.plate or "", {})
    if plate_info.get("stolen"):
        score += 50.0
    if plate_info.get("flagged"):
        score += 30.0
    return min(round(score, 2), 100.0)


def build_scene_summary(detections: List["DetectionResult"], alerts: List["AlertResult"], session_id: str) -> Dict:
    persons = [d for d in detections if d.det_type == "person"]
    vehicles = [d for d in detections if d.det_type == "vehicle"]
    flagged = [d for d in detections if d.is_flagged]
    pose_threats = [d for d in detections if d.pose_threat]
    masked = [d for d in detections if d.mask_detected]
    returning = [d for d in detections if d.is_returning]
    emotions = Counter(d.emotion for d in persons if d.emotion)
    speeds = [d.speed_kph for d in vehicles if d.speed_kph and d.speed_kph > 0]
    critical_alerts = [a for a in alerts if a.severity == "critical"]
    high_alerts = [a for a in alerts if a.severity == "high"]
    scene_threat = "none"
    if critical_alerts or flagged:
        scene_threat = "critical"
    elif high_alerts or pose_threats:
        scene_threat = "high"
    elif len(persons) >= CROWD_DENSITY_ALERT_THRESHOLD:
        scene_threat = "medium"
    return {
        "total_persons": len(persons),
        "total_vehicles": len(vehicles),
        "flagged_detected": len(flagged),
        "pose_threats": len(pose_threats),
        "masked_persons": len(masked),
        "returning_persons": len(returning),
        "dominant_emotion": emotions.most_common(1)[0][0] if emotions else "neutral",
        "emotion_breakdown": dict(emotions),
        "avg_vehicle_speed": round(float(np.mean(speeds)), 1) if speeds else 0.0,
        "max_vehicle_speed": round(max(speeds), 1) if speeds else 0.0,
        "critical_alerts": len(critical_alerts),
        "high_alerts": len(high_alerts),
        "scene_threat_level": scene_threat,
        "night_mode_active": night_mode_active.get(session_id, False),
    }


def generate_person_timeline(person_id: str) -> List[Dict]:
    rows = db_execute(
        """SELECT d.detected_at, d.session_id, d.emotion, d.speed_kph, d.pose_threat,
           d.behavior_flags, d.frame_snapshot_url, d.night_mode
           FROM detections d WHERE d.person_id = %s ORDER BY d.detected_at ASC""",
        (person_id,), fetch="all"
    )
    timeline = []
    for r in (rows or []):
        entry = dict(r)
        for k, v in entry.items():
            if isinstance(v, datetime.datetime):
                entry[k] = v.isoformat()
        timeline.append(entry)
    return timeline


def generate_vehicle_route(vehicle_id: str) -> List[Dict]:
    rows = db_execute(
        """SELECT pr.read_at, pr.speed_at_read, pr.session_id, pr.snapshot_url
           FROM plate_reads pr WHERE pr.vehicle_id = %s ORDER BY pr.read_at ASC""",
        (vehicle_id,), fetch="all"
    )
    route = []
    for r in (rows or []):
        entry = dict(r)
        for k, v in entry.items():
            if isinstance(v, datetime.datetime):
                entry[k] = v.isoformat()
        route.append(entry)
    return route


def build_demographic_report(session_id: str) -> Dict:
    counters = demographic_counters.get(session_id, Counter())
    gender_data = {k.replace("gender_", ""): v for k, v in counters.items() if k.startswith("gender_")}
    age_data = {k.replace("age_", ""): v for k, v in counters.items() if k.startswith("age_")}
    emotion_data = {k.replace("emotion_", ""): v for k, v in counters.items() if k.startswith("emotion_")}
    total = sum(gender_data.values()) or 1
    gender_pct = {k: round(v / total * 100, 1) for k, v in gender_data.items()}
    age_pct = {k: round(v / sum(age_data.values() or [1]) * 100, 1) for k, v in age_data.items()}
    emotion_pct = {k: round(v / sum(emotion_data.values() or [1]) * 100, 1) for k, v in emotion_data.items()}
    return {
        "session_id": session_id,
        "gender": {"counts": gender_data, "percentages": gender_pct},
        "age_ranges": {"counts": age_data, "percentages": age_pct},
        "emotions": {"counts": emotion_data, "percentages": emotion_pct},
        "total_persons_counted": total,
    }


def detect_convoy_pattern(session_id: str, vehicle_tracks: List[Dict]) -> Optional[Dict]:
    if len(vehicle_tracks) < 3:
        return None
    try:
        positions = [(v["cx"], v["cy"]) for v in vehicle_tracks]
        speeds = [v.get("speed", 0) for v in vehicle_tracks]
        speed_variance = float(np.var(speeds)) if speeds else 0
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        if speed_variance < 50 and (spread_x < 400 or spread_y < 400):
            return {
                "detected": True,
                "vehicle_count": len(vehicle_tracks),
                "speed_variance": round(speed_variance, 2),
                "spread_x": round(spread_x, 1),
                "spread_y": round(spread_y, 1),
            }
    except Exception:
        pass
    return None


def draw_trajectory_overlay(frame: np.ndarray, track_id: int, color: Tuple = (0, 255, 255)) -> np.ndarray:
    positions = track_position_histories.get(track_id, [])
    if len(positions) < 2:
        return frame
    for i in range(1, len(positions)):
        p1 = (int(positions[i-1][0]), int(positions[i-1][1]))
        p2 = (int(positions[i][0]), int(positions[i][1]))
        alpha = i / len(positions)
        c = tuple(int(c * alpha) for c in color)
        cv2.line(frame, p1, p2, c, 1, cv2.LINE_AA)
    future = compute_trajectory_prediction(track_id, 8)
    if future and positions:
        last = (int(positions[-1][0]), int(positions[-1][1]))
        for fx, fy in future:
            fp = (int(fx), int(fy))
            cv2.line(frame, last, fp, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.circle(frame, fp, 2, (200, 200, 200), -1)
            last = fp
    return frame


def extract_body_segments(frame: np.ndarray, bbox: List[float]) -> Dict[str, np.ndarray]:
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h = y2 - y1
    head_crop = frame[y1:y1 + h//4, x1:x2]
    torso_crop = frame[y1 + h//4:y1 + h*2//3, x1:x2]
    lower_crop = frame[y1 + h*2//3:y2, x1:x2]
    return {"head": head_crop, "torso": torso_crop, "lower": lower_crop}


def detect_weapon_in_hands(pose_keypoints: np.ndarray, frame: np.ndarray) -> bool:
    try:
        if pose_keypoints.shape[0] < 11:
            return False
        left_wrist = pose_keypoints[9][:2]
        right_wrist = pose_keypoints[10][:2]
        h, w = frame.shape[:2]
        for wrist in [left_wrist, right_wrist]:
            wx, wy = int(wrist[0]), int(wrist[1])
            r = 30
            x1 = max(0, wx - r)
            y1 = max(0, wy - r)
            x2 = min(w, wx + r)
            y2 = min(h, wy + r)
            if x2 <= x1 or y2 <= y1:
                continue
            hand_region = frame[y1:y2, x1:x2]
            if hand_region.size == 0:
                continue
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density > 0.22:
                return True
        return False
    except Exception:
        return False


def compute_scene_change_score(session_id: str, frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_sub = motion_background_subtractors.get(session_id)
    if bg_sub is None:
        return 0.0
    mask = bg_sub.apply(frame, learningRate=0)
    motion_ratio = float(np.sum(mask > 0)) / mask.size
    return round(motion_ratio * 100, 2)


def cluster_persons_by_proximity(detections: List["DetectionResult"], threshold_px: float = 120.0) -> List[List[int]]:
    persons = [(i, d) for i, d in enumerate(detections) if d.det_type == "person"]
    if not persons:
        return []
    clusters = []
    visited = set()
    for i, (idx_i, det_i) in enumerate(persons):
        if idx_i in visited:
            continue
        cluster = [idx_i]
        visited.add(idx_i)
        cx_i = (det_i.bbox[0] + det_i.bbox[2]) / 2
        cy_i = (det_i.bbox[1] + det_i.bbox[3]) / 2
        for j, (idx_j, det_j) in enumerate(persons):
            if idx_j in visited:
                continue
            cx_j = (det_j.bbox[0] + det_j.bbox[2]) / 2
            cy_j = (det_j.bbox[1] + det_j.bbox[3]) / 2
            dist = math.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
            if dist < threshold_px:
                cluster.append(idx_j)
                visited.add(idx_j)
        clusters.append(cluster)
    return clusters


def detect_group_behavior(clusters: List[List[int]], detections: List["DetectionResult"]) -> List[Dict]:
    events = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        members = [detections[i] for i in cluster if i < len(detections)]
        flagged_in_group = sum(1 for m in members if m.is_flagged)
        threat_in_group = sum(1 for m in members if m.pose_threat)
        emotions = [m.emotion for m in members if m.emotion]
        angry_count = emotions.count("angry")
        events.append({
            "cluster_size": len(cluster),
            "flagged_members": flagged_in_group,
            "threat_postures": threat_in_group,
            "angry_members": angry_count,
            "risk": "high" if flagged_in_group > 0 or threat_in_group > 1 else "medium" if angry_count > 1 else "low",
        })
    return events


def generate_shift_report(session_id: str, period_start: datetime.datetime, period_end: datetime.datetime) -> Dict:
    total_detections = db_execute(
        "SELECT COUNT(*) FROM detections WHERE session_id = %s AND detected_at BETWEEN %s AND %s",
        (session_id, period_start, period_end), fetch="scalar"
    ) or 0
    total_alerts = db_execute(
        "SELECT COUNT(*) FROM alerts WHERE session_id = %s AND created_at BETWEEN %s AND %s",
        (session_id, period_start, period_end), fetch="scalar"
    ) or 0
    critical_alerts = db_execute(
        "SELECT COUNT(*) FROM alerts WHERE session_id = %s AND severity = 'critical' AND created_at BETWEEN %s AND %s",
        (session_id, period_start, period_end), fetch="scalar"
    ) or 0
    unique_persons = db_execute(
        "SELECT COUNT(DISTINCT person_id) FROM detections WHERE session_id = %s AND detected_at BETWEEN %s AND %s AND person_id IS NOT NULL",
        (session_id, period_start, period_end), fetch="scalar"
    ) or 0
    unique_vehicles = db_execute(
        "SELECT COUNT(DISTINCT vehicle_id) FROM detections WHERE session_id = %s AND detected_at BETWEEN %s AND %s AND vehicle_id IS NOT NULL",
        (session_id, period_start, period_end), fetch="scalar"
    ) or 0
    top_alert_types = db_execute(
        "SELECT alert_type, COUNT(*) as c FROM alerts WHERE session_id = %s AND created_at BETWEEN %s AND %s GROUP BY alert_type ORDER BY c DESC LIMIT 5",
        (session_id, period_start, period_end), fetch="all"
    ) or []
    behavior_summary = db_execute(
        "SELECT behavior_type, COUNT(*) as c FROM behavior_events WHERE session_id = %s AND detected_at BETWEEN %s AND %s GROUP BY behavior_type ORDER BY c DESC",
        (session_id, period_start, period_end), fetch="all"
    ) or []
    demo = build_demographic_report(session_id)
    return {
        "session_id": session_id,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "total_detections": total_detections,
        "total_alerts": total_alerts,
        "critical_alerts": critical_alerts,
        "unique_persons": unique_persons,
        "unique_vehicles": unique_vehicles,
        "top_alert_types": [dict(r) for r in top_alert_types],
        "behavior_summary": [dict(r) for r in behavior_summary],
        "demographics": demo,
    }


def parse_alert_severity_from_score(score: float) -> str:
    if score >= 70:
        return "critical"
    elif score >= 50:
        return "high"
    elif score >= 25:
        return "medium"
    return "low"


def deduplicate_alerts(alerts: List["AlertResult"], window_seconds: float = 10.0) -> List["AlertResult"]:
    seen: Dict[str, float] = {}
    result = []
    now = time.time()
    for a in alerts:
        key = f"{a.alert_type}_{a.person_id}_{a.vehicle_id}_{a.track_id}"
        last = seen.get(key, 0)
        if now - last > window_seconds:
            result.append(a)
            seen[key] = now
    return result


def run_thermal_simulation(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    thermal = cv2.applyColorMap(blurred, cv2.COLORMAP_INFERNO)
    return thermal


def annotate_speed_color_band(frame: np.ndarray, x: int, y: int, speed: float) -> np.ndarray:
    if speed < 60:
        color = (0, 220, 0)
        label = "SAFE"
    elif speed < 100:
        color = (0, 165, 255)
        label = "WARN"
    else:
        color = (0, 0, 255)
        label = "OVER"
    cv2.putText(frame, f"{speed:.0f}km/h {label}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame


def build_plate_region_guess(plate: str) -> str:
    if not plate:
        return "unknown"
    if len(plate) == 7 and plate[:3].isalpha() and plate[3:].isdigit():
        return "UK_style"
    if len(plate) >= 5 and plate[:2].isdigit():
        return "EU_style"
    if len(plate) == 7 and plate[0].isalpha():
        return "US_style"
    return "generic"


def compute_occlusion_score(bbox: List[float], frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        return 0.5
    area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    ratio = area / frame_area
    if ratio < 0.005:
        return 0.8
    elif ratio > 0.4:
        return 0.3
    return 0.1


def reidentify_by_clothing(crop_a: np.ndarray, crop_b: np.ndarray) -> float:
    try:
        h_a = cv2.calcHist([crop_a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        h_b = cv2.calcHist([crop_b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(h_a, h_a)
        cv2.normalize(h_b, h_b)
        score = cv2.compareHist(h_a, h_b, cv2.HISTCMP_CORREL)
        return round(float(score), 4)
    except Exception:
        return 0.0


def batch_face_search(embeddings: List[np.ndarray]) -> List[Optional[Dict]]:
    return [match_face_embedding(e) for e in embeddings]


def compute_inter_person_distances(detections: List["DetectionResult"]) -> List[Dict]:
    persons = [d for d in detections if d.det_type == "person"]
    distances = []
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            a, b = persons[i], persons[j]
            cx_a = (a.bbox[0] + a.bbox[2]) / 2
            cy_a = (a.bbox[1] + a.bbox[3]) / 2
            cx_b = (b.bbox[0] + b.bbox[2]) / 2
            cy_b = (b.bbox[1] + b.bbox[3]) / 2
            dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
            distances.append({
                "person_a": a.person_id,
                "person_b": b.person_id,
                "distance_px": round(dist, 1),
                "both_flagged": a.is_flagged and b.is_flagged,
            })
    return distances


def compute_zone_statistics(session_id: str) -> Dict:
    intrusions = zone_intrusion_log.get(session_id, [])
    if not intrusions:
        return {"total_intrusions": 0, "unique_persons": 0, "zones_breached": 0}
    persons = set(i.get("person_id") for i in intrusions if i.get("person_id"))
    zones = set(tuple(i.get("zone", [])) for i in intrusions)
    return {
        "total_intrusions": len(intrusions),
        "unique_persons": len(persons),
        "zones_breached": len(zones),
        "latest": intrusions[-1] if intrusions else None,
    }


def adaptive_frame_rate_control(session: Dict, current_fps: float, alert_count: int) -> int:
    skip = FRAME_SKIP
    if alert_count > 5:
        skip = 1
    elif current_fps < 10:
        skip = 1
    elif current_fps < 20:
        skip = 2
    else:
        skip = FRAME_SKIP
    return skip


def emit_webhook_alert(alert: "AlertResult", session_id: str, webhook_url: Optional[str]):
    if not webhook_url:
        return
    try:
        import urllib.request as urlreq
        payload = json.dumps({
            "system": "ARGUS",
            "session_id": session_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "description": alert.description,
            "person_id": alert.person_id,
            "vehicle_id": alert.vehicle_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }).encode()
        req = urlreq.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        urlreq.urlopen(req, timeout=3)
    except Exception as e:
        logger.debug(f"Webhook delivery failed: {e}")


def generate_session_heatmap_image(session_id: str, frame_w: int = 1280, frame_h: int = 720) -> Optional[bytes]:
    if session_id not in attention_heatmap:
        return None
    hm = attention_heatmap[session_id]
    hm_resized = cv2.resize(hm, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    if np.max(hm_resized) == 0:
        return None
    hm_norm = (hm_resized / np.max(hm_resized) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".png", hm_color)
    return buf.tobytes()


def compute_speed_percentiles(vehicle_id: str) -> Dict:
    row = db_execute("SELECT speed_history FROM vehicles WHERE id = %s", (vehicle_id,), fetch="one")
    if not row or not row["speed_history"]:
        return {}
    history = row["speed_history"]
    if isinstance(history, str):
        history = json.loads(history)
    speeds = [entry["speed"] for entry in history if "speed" in entry]
    if not speeds:
        return {}
    arr = np.array(speeds)
    return {
        "p25": round(float(np.percentile(arr, 25)), 1),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "mean": round(float(np.mean(arr)), 1),
        "max": round(float(np.max(arr)), 1),
        "min": round(float(np.min(arr)), 1),
        "sample_count": len(speeds),
    }


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def face_quality_score(face_crop: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        score = 0.0
        if laplacian_var > 100:
            score += 40.0
        elif laplacian_var > 50:
            score += 20.0
        if 80 < brightness < 200:
            score += 30.0
        elif 50 < brightness < 220:
            score += 15.0
        h, w = gray.shape
        if h >= 80 and w >= 80:
            score += 30.0
        elif h >= 40 and w >= 40:
            score += 15.0
        return min(round(score, 1), 100.0)
    except Exception:
        return 0.0


def detect_hat_or_hood(face_crop: np.ndarray) -> bool:
    try:
        h, w = face_crop.shape[:2]
        top_strip = face_crop[:h // 5, :]
        hsv = cv2.cvtColor(top_strip, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
        dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
        return dark_ratio > 0.45
    except Exception:
        return False


def estimate_person_direction(track_id: int) -> str:
    positions = track_position_histories.get(track_id, [])
    if len(positions) < 3:
        return "stationary"
    dx = positions[-1][0] - positions[-3][0]
    dy = positions[-1][1] - positions[-3][1]
    mag = math.sqrt(dx ** 2 + dy ** 2)
    if mag < 5:
        return "stationary"
    angle = math.degrees(math.atan2(-dy, dx))
    if -45 <= angle < 45:
        return "right"
    elif 45 <= angle < 135:
        return "up"
    elif angle >= 135 or angle < -135:
        return "left"
    else:
        return "down"


def build_watch_proximity_check(person_id: str, detections: List["DetectionResult"]) -> List[Dict]:
    watch_rows = db_execute(
        "SELECT w.list_name, w.reason, p.id as wp_id FROM watchlist w JOIN persons p ON w.person_id = p.id WHERE w.active = TRUE",
        fetch="all"
    ) or []
    watch_ids = {str(r["wp_id"]) for r in watch_rows}
    proximity_alerts = []
    for det in detections:
        if det.person_id and det.person_id in watch_ids and det.person_id != person_id:
            entry = next((r for r in watch_rows if str(r["wp_id"]) == det.person_id), None)
            if entry:
                proximity_alerts.append({
                    "watchlist_person_id": det.person_id,
                    "list_name": entry["list_name"],
                    "reason": entry["reason"],
                    "bbox": det.bbox,
                })
    return proximity_alerts


def async_save_heatmap_snapshot(session_id: str):
    try:
        img_bytes = generate_session_heatmap_image(session_id)
        if img_bytes:
            key = f"heatmaps/{session_id}/{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            url = upload_bytes_to_r2(img_bytes, key, "image/png")
            db_execute(
                "INSERT INTO heatmap_snapshots (session_id, heatmap_type, image_url) VALUES (%s,%s,%s)",
                (session_id, "attention", url)
            )
    except Exception as e:
        logger.debug(f"Heatmap snapshot save failed: {e}")


def purge_stale_track_data(max_age_seconds: float = 120.0):
    now = time.time()
    stale_tracks = [tid for tid, t in track_last_seen.items() if now - t > max_age_seconds]
    for tid in stale_tracks:
        track_frame_counts.pop(tid, None)
        track_speed_histories.pop(tid, None)
        track_position_histories.pop(tid, None)
        track_first_seen.pop(tid, None)
        track_last_seen.pop(tid, None)


async def background_maintenance_loop():
    while True:
        try:
            await asyncio.sleep(60)
            purge_stale_track_data()
            stale_loiter = [k for k, v in loiter_tracker.items() if time.time() - v > 300]
            for k in stale_loiter:
                del loiter_tracker[k]
            stale_objects = [k for k, v in object_abandonment_tracker.items() if time.time() - v.get("first_seen", 0) > 600]
            for k in stale_objects:
                del object_abandonment_tracker[k]
            if active_sessions:
                for sid in list(active_sessions.keys()):
                    if active_sessions.get(sid, {}).get("enable_heatmap"):
                        executor.submit(async_save_heatmap_snapshot, sid)
            db_log_system("INFO", "maintenance", f"Maintenance run: purged {len(stale_tracks if False else stale_loiter)} stale entries")
        except Exception as e:
            logger.error(f"Maintenance loop error: {e}")


@app.on_event("startup")
async def start_maintenance():
    asyncio.create_task(background_maintenance_loop())


@app.get("/persons/{person_id}/timeline", tags=["persons"])
async def person_timeline(person_id: str, user=Depends(require_auth)):
    timeline = generate_person_timeline(person_id)
    return {"person_id": person_id, "events": timeline, "count": len(timeline)}


@app.get("/persons/{person_id}/risk", tags=["persons"])
async def person_risk_assessment(person_id: str, user=Depends(require_auth)):
    row = db_execute(
        "SELECT id, name, threat_level, threat_score, flagged, flag_reason, seen_count, dominant_emotion, gait_signature, behavior_tags FROM persons WHERE id = %s",
        (person_id,), fetch="one"
    )
    if not row:
        raise HTTPException(status_code=404, detail="Person not found")
    r = dict(row)
    alerts_count = db_execute(
        "SELECT COUNT(*) FROM alerts WHERE person_id = %s AND severity IN ('high','critical')",
        (person_id,), fetch="scalar"
    ) or 0
    intrusion_count = db_execute(
        "SELECT COUNT(*) FROM behavior_events WHERE person_id = %s AND behavior_type = 'restricted_zone_intrusion'",
        (person_id,), fetch="scalar"
    ) or 0
    base_score = r.get("threat_score") or 0.0
    if r.get("flagged"):
        base_score = max(base_score, 70.0)
    base_score += min(int(alerts_count) * 5.0, 20.0)
    base_score += min(int(intrusion_count) * 8.0, 16.0)
    base_score = min(base_score, 100.0)
    return {
        "person_id": person_id,
        "name": r.get("name"),
        "risk_score": round(base_score, 2),
        "risk_level": parse_alert_severity_from_score(base_score),
        "threat_level": r.get("threat_level"),
        "flagged": r.get("flagged"),
        "flag_reason": r.get("flag_reason"),
        "high_critical_alerts": int(alerts_count),
        "zone_intrusions": int(intrusion_count),
        "total_sightings": r.get("seen_count", 0),
        "dominant_emotion": r.get("dominant_emotion"),
    }


@app.get("/vehicles/{vehicle_id}/route", tags=["vehicles"])
async def vehicle_route(vehicle_id: str, user=Depends(require_auth)):
    route = generate_vehicle_route(vehicle_id)
    return {"vehicle_id": vehicle_id, "route": route, "stops": len(route)}


@app.get("/vehicles/{vehicle_id}/speed-stats", tags=["vehicles"])
async def vehicle_speed_stats(vehicle_id: str, user=Depends(require_auth)):
    return compute_speed_percentiles(vehicle_id)


@app.get("/sessions/{session_id}/report", tags=["sessions"])
async def session_shift_report(
    session_id: str,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    user=Depends(require_auth)
):
    try:
        ps = datetime.datetime.fromisoformat(period_start) if period_start else datetime.datetime.utcnow() - datetime.timedelta(hours=8)
        pe = datetime.datetime.fromisoformat(period_end) if period_end else datetime.datetime.utcnow()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO 8601.")
    return generate_shift_report(session_id, ps, pe)


@app.get("/sessions/{session_id}/scene-summary", tags=["sessions"])
async def session_scene_summary(session_id: str, user=Depends(require_auth)):
    s = active_sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not active")
    return {
        "session_id": session_id,
        "fps": s.get("fps", 0),
        "frame_count": s.get("frame_count", 0),
        "unique_persons": len(s.get("unique_persons", set())),
        "unique_vehicles": len(s.get("unique_vehicles", set())),
        "alert_count": s.get("alert_count", 0),
        "night_mode": night_mode_active.get(session_id, False),
        "zone_stats": compute_zone_statistics(session_id),
        "crowd_density": {
            "grid_shape": list(crowd_density_grid[session_id].shape) if session_id in crowd_density_grid else None,
            "total": int(np.sum(crowd_density_grid[session_id])) if session_id in crowd_density_grid else 0,
        },
        "heatmap_active": session_id in attention_heatmap,
    }


@app.get("/sessions/{session_id}/demographics-full", tags=["analytics"])
async def session_demographics_full(session_id: str, user=Depends(require_auth)):
    return build_demographic_report(session_id)


@app.post("/analyze/batch", tags=["analysis"])
async def batch_image_analysis(files: List[UploadFile] = File(...), user=Depends(require_auth)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 images per batch")
    results = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            results.append({"filename": file.filename, "error": "invalid image"})
            continue
        session_id = "batch_" + uuid.uuid4().hex[:8]
        loop = asyncio.get_event_loop()
        detections, alerts = await loop.run_in_executor(
            executor, process_full_frame,
            frame.copy(), session_id, 1, 30.0, None, [], False, False
        )
        results.append({
            "filename": file.filename,
            "detections": len(detections),
            "alerts": len(alerts),
            "persons": sum(1 for d in detections if d.det_type == "person"),
            "vehicles": sum(1 for d in detections if d.det_type == "vehicle"),
            "flagged": sum(1 for d in detections if d.is_flagged),
            "scene_summary": build_scene_summary(detections, alerts, session_id),
        })
    return {"batch_size": len(files), "results": results}


@app.get("/analyze/trajectory/{track_id}", tags=["analysis"])
async def get_trajectory(track_id: int, steps: int = 10, user=Depends(require_auth)):
    history = track_position_histories.get(track_id, [])
    future = compute_trajectory_prediction(track_id, steps)
    direction = estimate_person_direction(track_id)
    return {
        "track_id": track_id,
        "history": history,
        "predicted_path": future,
        "direction": direction,
        "frame_count": track_frame_counts.get(track_id, 0),
        "first_seen_ts": track_first_seen.get(track_id),
        "last_seen_ts": track_last_seen.get(track_id),
    }


@app.get("/analyze/heatmap/{session_id}/image", tags=["analysis"])
async def get_heatmap_image(session_id: str, user=Depends(require_auth)):
    img_bytes = generate_session_heatmap_image(session_id)
    if not img_bytes:
        raise HTTPException(status_code=404, detail="No heatmap data for this session")
    return Response(content=img_bytes, media_type="image/png")


@app.get("/analyze/speed/realtime", tags=["analysis"])
async def realtime_speed_data(user=Depends(require_auth)):
    data = {}
    for track_id, speed_hist in track_speed_histories.items():
        if speed_hist:
            data[track_id] = {
                "current": round(float(list(speed_hist)[-1]), 1),
                "avg": round(float(np.mean(list(speed_hist))), 1),
                "max": round(float(np.max(list(speed_hist))), 1),
                "samples": len(speed_hist),
            }
    return {"tracks": data, "count": len(data)}


@app.post("/watchlist/bulk-add", tags=["watchlist"])
async def bulk_add_to_watchlist(entries: List[WatchlistAddRequest], user=Depends(require_auth)):
    added = []
    for req in entries:
        entry_id = str(uuid.uuid4())
        db_execute(
            "INSERT INTO watchlist (id, list_name, list_type, person_id, vehicle_plate, reason, priority, added_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (entry_id, req.list_name, req.list_type, req.person_id, req.vehicle_plate, req.reason, req.priority, req.added_by)
        )
        added.append(entry_id)
    return {"status": "added", "count": len(added), "ids": added}


@app.get("/watchlist/lists", tags=["watchlist"])
async def get_watchlist_names(user=Depends(require_auth)):
    rows = db_execute(
        "SELECT DISTINCT list_name, COUNT(*) as count FROM watchlist WHERE active = TRUE GROUP BY list_name ORDER BY list_name",
        fetch="all"
    )
    return [dict(r) for r in (rows or [])]


@app.get("/stats/hourly-detections", tags=["analytics"])
async def hourly_detection_breakdown(session_id: Optional[str] = None, user=Depends(require_auth)):
    if session_id:
        rows = db_execute(
            """SELECT DATE_TRUNC('hour', detected_at) as hour, detection_type, COUNT(*) as count
               FROM detections WHERE session_id = %s
               GROUP BY hour, detection_type ORDER BY hour DESC LIMIT 168""",
            (session_id,), fetch="all"
        )
    else:
        rows = db_execute(
            """SELECT DATE_TRUNC('hour', detected_at) as hour, detection_type, COUNT(*) as count
               FROM detections WHERE detected_at > NOW() - INTERVAL '7 days'
               GROUP BY hour, detection_type ORDER BY hour DESC""",
            fetch="all"
        )
    result = []
    for r in (rows or []):
        entry = dict(r)
        if isinstance(entry.get("hour"), datetime.datetime):
            entry["hour"] = entry["hour"].isoformat()
        result.append(entry)
    return result


@app.get("/stats/alert-timeline", tags=["analytics"])
async def alert_timeline(days: int = 7, user=Depends(require_auth)):
    rows = db_execute(
        """SELECT DATE_TRUNC('hour', created_at) as hour, severity, COUNT(*) as count
           FROM alerts WHERE created_at > NOW() - INTERVAL '%s days'
           GROUP BY hour, severity ORDER BY hour DESC""",
        (days,), fetch="all"
    )
    result = []
    for r in (rows or []):
        entry = dict(r)
        if isinstance(entry.get("hour"), datetime.datetime):
            entry["hour"] = entry["hour"].isoformat()
        result.append(entry)
    return result


@app.get("/stats/persons/emotion-breakdown", tags=["analytics"])
async def emotion_breakdown_global(user=Depends(require_auth)):
    rows = db_execute(
        "SELECT dominant_emotion, COUNT(*) as count FROM persons WHERE dominant_emotion IS NOT NULL GROUP BY dominant_emotion ORDER BY count DESC",
        fetch="all"
    )
    return [dict(r) for r in (rows or [])]


@app.get("/stats/persons/age-gender", tags=["analytics"])
async def age_gender_breakdown(user=Depends(require_auth)):
    rows = db_execute(
        "SELECT age_range, gender, COUNT(*) as count FROM persons WHERE age_range IS NOT NULL AND gender IS NOT NULL GROUP BY age_range, gender ORDER BY age_range",
        fetch="all"
    )
    return [dict(r) for r in (rows or [])]


@app.get("/stats/flagged-summary", tags=["analytics"])
async def flagged_summary(user=Depends(require_auth)):
    flagged_persons = db_execute("SELECT COUNT(*) FROM persons WHERE flagged = TRUE", fetch="scalar") or 0
    flagged_vehicles = db_execute("SELECT COUNT(*) FROM vehicles WHERE flagged = TRUE", fetch="scalar") or 0
    stolen_vehicles = db_execute("SELECT COUNT(*) FROM vehicles WHERE stolen = TRUE", fetch="scalar") or 0
    flagged_plates_read = db_execute("SELECT COUNT(*) FROM plate_reads WHERE flagged = TRUE", fetch="scalar") or 0
    recent_flagged_detections = db_execute(
        "SELECT COUNT(*) FROM detections WHERE is_flagged = TRUE AND detected_at > NOW() - INTERVAL '24 hours'",
        fetch="scalar"
    ) or 0
    return {
        "flagged_persons": flagged_persons,
        "flagged_vehicles": flagged_vehicles,
        "stolen_vehicles": stolen_vehicles,
        "flagged_plate_reads": flagged_plates_read,
        "flagged_detections_24h": recent_flagged_detections,
    }


@app.post("/admin/reindex/faces", tags=["admin"])
async def reindex_face_database(user=Depends(require_admin)):
    load_known_faces_from_db()
    count = len(known_face_embeddings)
    flagged = len(flagged_faces_cache)
    db_log_system("INFO", "admin", f"Face DB reindexed: {count} faces, {flagged} flagged")
    return {"status": "reindexed", "total_faces": count, "flagged_faces": flagged}


@app.post("/admin/sessions/terminate-all", tags=["admin"])
async def terminate_all_sessions(user=Depends(require_admin)):
    count = len(active_sessions)
    for sid in list(active_sessions.keys()):
        active_sessions[sid]["running"] = False
        s = active_sessions[sid]
        db_update_session_stats(sid, {
            **s,
            "unique_persons": len(s.get("unique_persons", set())),
            "unique_vehicles": len(s.get("unique_vehicles", set())),
        })
        db_end_session(sid, time.time() - s.get("started_at", time.time()))
        del active_sessions[sid]
    db_log_system("WARN", "admin", f"All sessions terminated: {count}")
    return {"status": "terminated", "sessions_killed": count}


@app.delete("/admin/purge/detections", tags=["admin"])
async def purge_old_detections(older_than_days: int = 30, user=Depends(require_admin)):
    db_execute(
        "DELETE FROM detections WHERE detected_at < NOW() - INTERVAL '%s days'",
        (older_than_days,)
    )
    db_log_system("WARN", "admin", f"Purged detections older than {older_than_days} days")
    return {"status": "purged", "older_than_days": older_than_days}


@app.delete("/admin/purge/alerts", tags=["admin"])
async def purge_old_alerts(older_than_days: int = 60, user=Depends(require_admin)):
    db_execute(
        "DELETE FROM alerts WHERE created_at < NOW() - INTERVAL '%s days' AND resolved = TRUE",
        (older_than_days,)
    )
    return {"status": "purged", "older_than_days": older_than_days}


@app.get("/admin/db/stats", tags=["admin"])
async def database_table_stats(user=Depends(require_admin)):
    tables = ["persons", "vehicles", "detections", "alerts", "plate_reads", "audio_events", "behavior_events", "sessions", "watchlist", "gait_profiles", "system_logs"]
    stats = {}
    for table in tables:
        try:
            count = db_execute(f"SELECT COUNT(*) FROM {table}", fetch="scalar")
            stats[table] = int(count or 0)
        except Exception:
            stats[table] = -1
    return {"tables": stats}


@app.get("/admin/memory/stats", tags=["admin"])
async def memory_stats(user=Depends(require_admin)):
    return {
        "known_faces": len(known_face_embeddings),
        "flagged_cache": len(flagged_faces_cache),
        "known_plates": len(known_plate_registry),
        "active_sessions": len(active_sessions),
        "track_ids_tracked": len(track_position_histories),
        "loiter_keys": len(loiter_tracker),
        "audio_buffer": len(audio_event_buffer),
        "heatmap_sessions": len(attention_heatmap),
        "crowd_sessions": len(crowd_density_grid),
        "optical_flow_sessions": len(optical_flow_prev),
        "bg_subtractors": len(motion_background_subtractors),
        "zone_intrusion_logs": sum(len(v) for v in zone_intrusion_log.values()),
        "fight_buffers": len(fight_motion_buffer),
        "gait_store_sessions": len(gait_signature_store),
        "abandoned_objects": len(object_abandonment_tracker),
        "demographic_sessions": len(demographic_counters),
    }


@app.get("/export/session-report/{session_id}", tags=["export"])
async def export_session_report_json(session_id: str, user=Depends(require_auth)):
    session_row = db_execute("SELECT * FROM sessions WHERE id = %s", (session_id,), fetch="one")
    if not session_row:
        raise HTTPException(status_code=404, detail="Session not found")
    session_data = dict(session_row)
    for k, v in session_data.items():
        if isinstance(v, datetime.datetime):
            session_data[k] = v.isoformat()
    detections = db_execute(
        "SELECT id, detected_at, detection_type, person_id, vehicle_id, emotion, threat_score, speed_kph, plate_number FROM detections WHERE session_id = %s ORDER BY detected_at DESC",
        (session_id,), fetch="all"
    ) or []
    alerts = db_execute(
        "SELECT id, alert_type, severity, created_at, description, resolved FROM alerts WHERE session_id = %s ORDER BY created_at DESC",
        (session_id,), fetch="all"
    ) or []
    report = {
        "session": session_data,
        "detection_count": len(detections),
        "alert_count": len(alerts),
        "detections": [dict(d) for d in detections],
        "alerts": [dict(a) for a in alerts],
        "demographics": build_demographic_report(session_id),
        "zone_stats": compute_zone_statistics(session_id),
        "generated_at": datetime.datetime.utcnow().isoformat(),
    }

    def default_serializer(obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return str(obj)

    report_json = json.dumps(report, default=default_serializer, indent=2)
    return StreamingResponse(
        io.BytesIO(report_json.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=argus_session_{session_id[:8]}.json"}
    )


@app.get("/export/heatmap/{session_id}/png", tags=["export"])
async def export_heatmap_png(session_id: str, user=Depends(require_auth)):
    img_bytes = generate_session_heatmap_image(session_id)
    if not img_bytes:
        raise HTTPException(status_code=404, detail="No heatmap available")
    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=argus_heatmap_{session_id[:8]}.png"}
    )


@app.get("/export/behavior/csv", tags=["export"])
async def export_behavior_csv(session_id: Optional[str] = None, user=Depends(require_auth)):
    if session_id:
        rows = db_execute(
            "SELECT id, session_id, detected_at, behavior_type, track_id, duration_seconds, confidence FROM behavior_events WHERE session_id = %s ORDER BY detected_at DESC",
            (session_id,), fetch="all"
        )
    else:
        rows = db_execute(
            "SELECT id, session_id, detected_at, behavior_type, track_id, duration_seconds, confidence FROM behavior_events ORDER BY detected_at DESC LIMIT 5000",
            fetch="all"
        )
    if not rows:
        return Response(content="", media_type="text/csv")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()

    def safe_row(r):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        return d

    writer.writerows([safe_row(r) for r in rows])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.read().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=argus_behavior.csv"}
    )


@app.get("/system/capabilities", tags=["system"])
async def system_capabilities():
    return {
        "ai_models": {
            "yolo_detector": yolo_detector is not None,
            "yolo_pose": yolo_pose_detector is not None,
            "yolo_segmentor": yolo_segmentor is not None,
            "face_analyzer": face_analyzer is not None,
            "ocr_engine": ocr_engine is not None,
        },
        "features": {
            "face_recognition": True,
            "emotion_detection": True,
            "age_gender_estimation": True,
            "gait_analysis": True,
            "vehicle_detection": True,
            "plate_ocr": True,
            "speed_tracking": True,
            "pose_threat_analysis": True,
            "crowd_density": True,
            "behavior_analysis": True,
            "audio_threat_detection": True,
            "night_mode": True,
            "attention_heatmap": True,
            "loitering_detection": True,
            "fight_detection": True,
            "abandoned_object": True,
            "tailgating_detection": True,
            "optical_flow": True,
            "demographic_analytics": True,
            "trajectory_prediction": True,
            "re_identification": True,
            "convoy_detection": True,
            "group_behavior": True,
            "thermal_simulation": True,
            "clothing_color_analysis": True,
            "accessory_detection": True,
            "height_estimation": True,
            "watchlist_management": True,
            "restricted_zone_enforcement": True,
            "multi_session_support": True,
            "websocket_streaming": True,
            "audio_streaming": True,
            "r2_snapshot_storage": True,
            "csv_export": True,
            "json_report_export": True,
        },
        "limits": {
            "max_sessions": MAX_SESSIONS,
            "max_workers": MAX_WORKERS,
            "frame_skip": FRAME_SKIP,
            "max_plausible_kph": MAX_PLAUSIBLE_KPH,
            "face_similarity_threshold": FACE_SIMILARITY_THRESHOLD,
            "loiter_seconds": LOITER_TIME_SECONDS,
            "crowd_alert_threshold": CROWD_DENSITY_ALERT_THRESHOLD,
        }
    }


@app.get("/system/routes", tags=["system"])
async def list_all_routes():
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name,
            })
    return {"routes": routes, "count": len(routes)}



def compute_multi_camera_overlap(session_ids: List[str], person_id: str) -> List[Dict]:
    results = []
    for sid in session_ids:
        rows = db_execute(
            "SELECT detected_at, frame_snapshot_url FROM detections WHERE session_id = %s AND person_id = %s ORDER BY detected_at DESC LIMIT 5",
            (sid, person_id), fetch="all"
        )
        if rows:
            results.append({
                "session_id": sid,
                "sightings": len(rows),
                "last_seen": rows[0]["detected_at"].isoformat() if isinstance(rows[0]["detected_at"], datetime.datetime) else str(rows[0]["detected_at"]),
            })
    return results


def compute_hotspot_zones(session_id: str) -> List[Dict]:
    if session_id not in crowd_density_grid:
        return []
    grid = crowd_density_grid[session_id]
    hotspots = []
    threshold = max(1, int(np.max(grid) * 0.6))
    for gy in range(grid.shape[0]):
        for gx in range(grid.shape[1]):
            if grid[gy, gx] >= threshold:
                hotspots.append({
                    "grid_x": gx,
                    "grid_y": gy,
                    "pixel_x": gx * HEATMAP_GRID_SIZE,
                    "pixel_y": gy * HEATMAP_GRID_SIZE,
                    "density": int(grid[gy, gx]),
                })
    hotspots.sort(key=lambda x: x["density"], reverse=True)
    return hotspots[:10]


def detect_perimeter_breach(session_id: str, detections: List["DetectionResult"], perimeter: List[Tuple]) -> List[Dict]:
    breaches = []
    if len(perimeter) < 3:
        return breaches
    perimeter_np = np.array(perimeter, dtype=np.float32)
    for det in detections:
        if det.det_type != "person":
            continue
        cx = (det.bbox[0] + det.bbox[2]) / 2
        cy = (det.bbox[1] + det.bbox[3]) / 2
        pt = (float(cx), float(cy))
        inside = cv2.pointPolygonTest(perimeter_np, pt, False)
        if inside >= 0:
            breaches.append({
                "person_id": det.person_id,
                "track_id": det.track_id,
                "position": {"x": cx, "y": cy},
                "flagged": det.is_flagged,
            })
    return breaches


def track_cross_session_person(person_id: str) -> Dict:
    sessions_seen = db_execute(
        "SELECT DISTINCT session_id FROM detections WHERE person_id = %s",
        (person_id,), fetch="all"
    ) or []
    total = db_execute(
        "SELECT COUNT(*), MIN(detected_at), MAX(detected_at) FROM detections WHERE person_id = %s",
        (person_id,), fetch="one"
    )
    return {
        "person_id": person_id,
        "sessions_seen": [r["session_id"] for r in sessions_seen],
        "total_sessions": len(sessions_seen),
        "total_detections": int(total[0]) if total else 0,
        "first_detected": total[1].isoformat() if total and total[1] else None,
        "last_detected": total[2].isoformat() if total and total[2] else None,
    }


def build_alert_correlation_matrix(session_id: str) -> Dict:
    rows = db_execute(
        "SELECT alert_type, COUNT(*) as c FROM alerts WHERE session_id = %s GROUP BY alert_type",
        (session_id,), fetch="all"
    ) or []
    alert_counts = {r["alert_type"]: int(r["c"]) for r in rows}
    correlations = {}
    for at in alert_counts:
        correlations[at] = {
            "count": alert_counts[at],
            "co_occurring": [],
        }
    return {"session_id": session_id, "alert_matrix": correlations}


def build_speed_heatmap_data(session_id: Optional[str] = None) -> List[Dict]:
    query = "SELECT speed_at_read, plate_number, read_at FROM plate_reads WHERE speed_at_read > 0"
    params = []
    if session_id:
        query += " AND session_id = %s"
        params.append(session_id)
    query += " ORDER BY read_at DESC LIMIT 500"
    rows = db_execute(query, tuple(params) if params else None, fetch="all") or []
    return [
        {
            "plate": r["plate_number"],
            "speed": float(r["speed_at_read"]),
            "ts": r["read_at"].isoformat() if isinstance(r["read_at"], datetime.datetime) else str(r["read_at"]),
        }
        for r in rows
    ]


def infer_person_intent(det: "DetectionResult", session_id: str) -> str:
    flags = det.behavior_flags or []
    if "restricted_zone" in flags:
        return "unauthorized_access"
    if "loitering" in flags:
        return "surveillance_or_waiting"
    if "running" in flags and det.pose_threat:
        return "fleeing_or_attacking"
    if det.pose_threat:
        return "confrontational"
    if det.mask_detected and det.is_returning:
        return "deliberate_concealment"
    if det.gait_label == "aggressive":
        return "aggressive_approach"
    if det.emotion in ["angry", "fear"]:
        return f"elevated_emotional_state_{det.emotion}"
    return "normal"


def score_frame_urgency(detections: List["DetectionResult"], alerts: List["AlertResult"]) -> float:
    score = 0.0
    for a in alerts:
        s = {"low": 5, "medium": 15, "high": 30, "critical": 60}.get(a.severity, 0)
        score += s
    for d in detections:
        if d.is_flagged:
            score += 25
        if d.pose_threat:
            score += 20
        if d.threat_level == "critical":
            score += 30
        elif d.threat_level == "high":
            score += 15
    return min(round(score, 2), 100.0)


def build_full_person_dossier(person_id: str) -> Dict:
    person = db_execute("SELECT * FROM persons WHERE id = %s", (person_id,), fetch="one")
    if not person:
        return {}
    p = dict(person)
    p.pop("embedding", None)
    for k, v in p.items():
        if isinstance(v, datetime.datetime):
            p[k] = v.isoformat()
    timeline = generate_person_timeline(person_id)
    cross_session = track_cross_session_person(person_id)
    gait = db_execute("SELECT * FROM gait_profiles WHERE person_id = %s ORDER BY captured_at DESC LIMIT 5", (person_id,), fetch="all") or []
    watchlist = db_execute("SELECT * FROM watchlist WHERE person_id = %s AND active = TRUE", (person_id,), fetch="all") or []
    alerts_list = db_execute("SELECT * FROM alerts WHERE person_id = %s ORDER BY created_at DESC LIMIT 20", (person_id,), fetch="all") or []
    associates = db_execute(
        "SELECT p2.id, p2.name, p2.threat_level, p2.flagged FROM persons p1 JOIN persons p2 ON p2.id::text = ANY(SELECT jsonb_array_elements(p1.known_associates)->>'id' FROM persons WHERE id = %s) WHERE p1.id = %s LIMIT 10",
        (person_id, person_id), fetch="all"
    ) or []

    def safe(rows):
        result = []
        for r in rows:
            d = dict(r)
            for k, v in d.items():
                if isinstance(v, datetime.datetime):
                    d[k] = v.isoformat()
            d.pop("embedding", None)
            result.append(d)
        return result

    return {
        "person": p,
        "cross_session_tracking": cross_session,
        "timeline": timeline,
        "gait_profiles": safe(gait),
        "watchlist_entries": safe(watchlist),
        "alerts": safe(alerts_list),
        "known_associates": safe(associates),
        "intent_inference": infer_person_intent(
            DetectionResult(
                det_type="person", bbox=[0,0,0,0], confidence=1.0,
                person_id=person_id,
                is_flagged=p.get("flagged", False),
                threat_level=p.get("threat_level", "none"),
                mask_detected=p.get("mask_detected", False),
                emotion=p.get("dominant_emotion"),
            ),
            "dossier"
        ),
    }


def build_full_vehicle_dossier(vehicle_id: str) -> Dict:
    vehicle = db_execute("SELECT * FROM vehicles WHERE id = %s", (vehicle_id,), fetch="one")
    if not vehicle:
        return {}
    v = dict(vehicle)
    for k, val in v.items():
        if isinstance(val, datetime.datetime):
            v[k] = val.isoformat()
    route = generate_vehicle_route(vehicle_id)
    speed_stats = compute_speed_percentiles(vehicle_id)
    plate_reads = db_execute("SELECT * FROM plate_reads WHERE vehicle_id = %s ORDER BY read_at DESC LIMIT 30", (vehicle_id,), fetch="all") or []
    linked_alerts = db_execute("SELECT * FROM alerts WHERE vehicle_id = %s ORDER BY created_at DESC LIMIT 20", (vehicle_id,), fetch="all") or []

    def safe(rows):
        result = []
        for r in rows:
            d = dict(r)
            for k, val in d.items():
                if isinstance(val, datetime.datetime):
                    d[k] = val.isoformat()
            result.append(d)
        return result

    return {
        "vehicle": v,
        "speed_statistics": speed_stats,
        "route": route,
        "plate_reads": safe(plate_reads),
        "alerts": safe(linked_alerts),
    }


@app.get("/persons/{person_id}/dossier", tags=["persons"])
async def get_person_dossier(person_id: str, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    dossier = await loop.run_in_executor(executor, build_full_person_dossier, person_id)
    if not dossier:
        raise HTTPException(status_code=404, detail="Person not found")
    return dossier


@app.get("/vehicles/{vehicle_id}/dossier", tags=["vehicles"])
async def get_vehicle_dossier(vehicle_id: str, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    dossier = await loop.run_in_executor(executor, build_full_vehicle_dossier, vehicle_id)
    if not dossier:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return dossier


@app.get("/persons/{person_id}/cross-session", tags=["persons"])
async def person_cross_session(person_id: str, user=Depends(require_auth)):
    return track_cross_session_person(person_id)


@app.get("/sessions/{session_id}/hotspots", tags=["sessions"])
async def session_hotspots(session_id: str, user=Depends(require_auth)):
    return {"session_id": session_id, "hotspots": compute_hotspot_zones(session_id)}


@app.post("/analyze/group", tags=["analysis"])
async def analyze_group_behavior(file: UploadFile = File(...), user=Depends(require_auth)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    session_id = "grp_" + uuid.uuid4().hex[:8]
    loop = asyncio.get_event_loop()
    detections, alerts = await loop.run_in_executor(
        executor, process_full_frame,
        frame.copy(), session_id, 1, 30.0, None, [], False, False
    )
    clusters = cluster_persons_by_proximity(detections)
    group_events = detect_group_behavior(clusters, detections)
    distances = compute_inter_person_distances(detections)
    urgency = score_frame_urgency(detections, alerts)
    return {
        "session_id": session_id,
        "total_persons": sum(1 for d in detections if d.det_type == "person"),
        "clusters": clusters,
        "group_behavior": group_events,
        "inter_person_distances": distances,
        "frame_urgency_score": urgency,
        "alerts": [asdict(a) for a in alerts],
    }


@app.get("/analyze/speed-heatmap", tags=["analysis"])
async def speed_heatmap_data(session_id: Optional[str] = None, user=Depends(require_auth)):
    return {"data": build_speed_heatmap_data(session_id)}


@app.get("/sessions/{session_id}/alert-correlation", tags=["analytics"])
async def session_alert_correlation(session_id: str, user=Depends(require_auth)):
    return build_alert_correlation_matrix(session_id)


@app.get("/persons/{person_id}/intent", tags=["persons"])
async def infer_intent(person_id: str, user=Depends(require_auth)):
    row = db_execute(
        "SELECT flagged, threat_level, mask_detected, dominant_emotion FROM persons WHERE id = %s",
        (person_id,), fetch="one"
    )
    if not row:
        raise HTTPException(status_code=404, detail="Person not found")
    det = DetectionResult(
        det_type="person", bbox=[0,0,0,0], confidence=1.0,
        person_id=person_id,
        is_flagged=row["flagged"] or False,
        threat_level=row["threat_level"] or "none",
        mask_detected=row["mask_detected"] or False,
        emotion=row["dominant_emotion"],
    )
    return {"person_id": person_id, "inferred_intent": infer_person_intent(det, "api")}


@app.get("/analyze/frame-urgency/{session_id}", tags=["analysis"])
async def frame_urgency(session_id: str, user=Depends(require_auth)):
    s = active_sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not active")
    return {
        "session_id": session_id,
        "alert_count": s.get("alert_count", 0),
        "fps": s.get("fps", 0),
        "unique_persons": len(s.get("unique_persons", set())),
    }


@app.get("/persons/flagged/recent", tags=["persons"])
async def recently_flagged_persons(limit: int = 20, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, name, flag_reason, flag_timestamp, threat_level, snapshot_url FROM persons WHERE flagged = TRUE ORDER BY flag_timestamp DESC NULLS LAST LIMIT %s",
        (limit,), fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.get("/vehicles/flagged/recent", tags=["vehicles"])
async def recently_flagged_vehicles(limit: int = 20, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, plate_number, flag_reason, flag_timestamp, stolen, vehicle_type, snapshot_url FROM vehicles WHERE flagged = TRUE ORDER BY flag_timestamp DESC NULLS LAST LIMIT %s",
        (limit,), fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.get("/alerts/unresolved/count", tags=["alerts"])
async def unresolved_alert_count(user=Depends(require_auth)):
    total = db_execute("SELECT COUNT(*) FROM alerts WHERE resolved = FALSE", fetch="scalar") or 0
    by_severity = db_execute(
        "SELECT severity, COUNT(*) as c FROM alerts WHERE resolved = FALSE GROUP BY severity",
        fetch="all"
    ) or []
    return {
        "total_unresolved": int(total),
        "by_severity": {r["severity"]: int(r["c"]) for r in by_severity},
    }


@app.get("/stats/persons/returning-rate", tags=["analytics"])
async def returning_person_rate(user=Depends(require_auth)):
    total = db_execute("SELECT COUNT(*) FROM persons", fetch="scalar") or 1
    returning = db_execute("SELECT COUNT(*) FROM persons WHERE seen_count > 1", fetch="scalar") or 0
    rate = round(int(returning) / int(total) * 100, 2)
    return {"total_persons": int(total), "returning_persons": int(returning), "return_rate_pct": rate}


@app.get("/stats/vehicles/speed-distribution", tags=["analytics"])
async def vehicle_speed_distribution(user=Depends(require_auth)):
    rows = db_execute(
        """SELECT
            SUM(CASE WHEN speed_at_read < 60 THEN 1 ELSE 0 END) as safe,
            SUM(CASE WHEN speed_at_read >= 60 AND speed_at_read < 100 THEN 1 ELSE 0 END) as warning,
            SUM(CASE WHEN speed_at_read >= 100 THEN 1 ELSE 0 END) as over_limit,
            AVG(speed_at_read) as avg_speed,
            MAX(speed_at_read) as max_speed
           FROM plate_reads WHERE speed_at_read > 0""",
        fetch="one"
    )
    if not rows:
        return {}
    return dict(rows)


@app.get("/live/flagged-alerts", tags=["live"])
async def live_flagged_alerts(user=Depends(require_auth)):
    rows = db_execute(
        "SELECT * FROM alerts WHERE severity IN ('high','critical') AND resolved = FALSE ORDER BY created_at DESC LIMIT 30",
        fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.get("/live/active-tracks", tags=["live"])
async def live_active_tracks(user=Depends(require_auth)):
    now = time.time()
    active = []
    for tid, last_t in track_last_seen.items():
        if now - last_t < 10.0:
            pos = track_position_histories.get(tid, [])
            active.append({
                "track_id": tid,
                "last_seen_ago_s": round(now - last_t, 2),
                "position": pos[-1] if pos else None,
                "frame_count": track_frame_counts.get(tid, 0),
                "current_speed": round(float(list(track_speed_histories[tid])[-1]), 1) if track_speed_histories.get(tid) else None,
            })
    return {"active_tracks": active, "count": len(active)}


@app.websocket("/ws/dashboard")
async def websocket_dashboard_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(3)
            active_track_count = sum(1 for tid, t in track_last_seen.items() if time.time() - t < 15)
            unresolved = db_execute("SELECT COUNT(*) FROM alerts WHERE resolved = FALSE", fetch="scalar") or 0
            critical = db_execute("SELECT COUNT(*) FROM alerts WHERE resolved = FALSE AND severity = 'critical'", fetch="scalar") or 0
            recent_persons = db_execute("SELECT COUNT(*) FROM detections WHERE detected_at > NOW() - INTERVAL '1 minute'", fetch="scalar") or 0
            await websocket.send_json({
                "event": "dashboard_tick",
                "ts": datetime.datetime.utcnow().isoformat(),
                "active_sessions": len(active_sessions),
                "active_ws_streams": len(active_ws_connections),
                "active_audio_streams": len(audio_ws_connections),
                "active_tracks": active_track_count,
                "known_faces": len(known_face_embeddings),
                "flagged_faces": len(flagged_faces_cache),
                "known_plates": len(known_plate_registry),
                "unresolved_alerts": int(unresolved),
                "critical_alerts": int(critical),
                "detections_last_minute": int(recent_persons),
                "audio_buffer_size": len(audio_event_buffer),
                "night_mode_sessions": sum(1 for v in night_mode_active.values() if v),
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Dashboard WS error: {e}")


@app.websocket("/ws/persons/live")
async def websocket_live_persons(websocket: WebSocket):
    await websocket.accept()
    try:
        last_ts = datetime.datetime.utcnow() - datetime.timedelta(seconds=5)
        while True:
            await asyncio.sleep(2)
            rows = db_execute(
                "SELECT id, name, seen_count, last_seen, flagged, threat_level, snapshot_url FROM persons WHERE last_seen > %s ORDER BY last_seen DESC LIMIT 20",
                (last_ts,), fetch="all"
            ) or []
            if rows:
                last_ts = datetime.datetime.utcnow()
                payload = []
                for r in rows:
                    d = dict(r)
                    for k, v in d.items():
                        if isinstance(v, datetime.datetime):
                            d[k] = v.isoformat()
                    payload.append(d)
                await websocket.send_json({"event": "persons_update", "persons": payload, "count": len(payload)})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Live persons WS error: {e}")


@app.websocket("/ws/vehicles/live")
async def websocket_live_vehicles(websocket: WebSocket):
    await websocket.accept()
    try:
        last_ts = datetime.datetime.utcnow() - datetime.timedelta(seconds=5)
        while True:
            await asyncio.sleep(2)
            rows = db_execute(
                "SELECT id, plate_number, vehicle_type, vehicle_color, avg_speed_kph, max_speed_kph, flagged, stolen, last_seen, snapshot_url FROM vehicles WHERE last_seen > %s ORDER BY last_seen DESC LIMIT 20",
                (last_ts,), fetch="all"
            ) or []
            if rows:
                last_ts = datetime.datetime.utcnow()
                payload = []
                for r in rows:
                    d = dict(r)
                    for k, v in d.items():
                        if isinstance(v, datetime.datetime):
                            d[k] = v.isoformat()
                    payload.append(d)
                await websocket.send_json({"event": "vehicles_update", "vehicles": payload, "count": len(payload)})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Live vehicles WS error: {e}")


@app.get("/persons/search/name", tags=["persons"])
async def search_person_by_name(q: str = Query(..., min_length=1), user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, name, alias, age_estimate, gender, threat_level, flagged, snapshot_url, seen_count FROM persons WHERE name ILIKE %s OR alias ILIKE %s ORDER BY seen_count DESC LIMIT 20",
        (f"%{q}%", f"%{q}%"), fetch="all"
    )
    return [dict(r) for r in (rows or [])]


@app.get("/persons/search/threat", tags=["persons"])
async def search_persons_by_threat(level: str = "high", user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, name, alias, threat_level, threat_score, flagged, last_seen, snapshot_url FROM persons WHERE threat_level = %s ORDER BY threat_score DESC LIMIT 50",
        (level,), fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.get("/persons/unknown/recent", tags=["persons"])
async def recent_unknown_persons(limit: int = 50, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, age_estimate, gender, first_seen, last_seen, seen_count, snapshot_url, mask_detected FROM persons WHERE name IS NULL ORDER BY last_seen DESC LIMIT %s",
        (limit,), fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.get("/vehicles/unregistered/recent", tags=["vehicles"])
async def recent_unregistered_vehicles(limit: int = 50, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, plate_number, vehicle_type, vehicle_color, first_seen, last_seen, seen_count, snapshot_url FROM vehicles WHERE owner_person_id IS NULL ORDER BY last_seen DESC LIMIT %s",
        (limit,), fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.post("/vehicles/{vehicle_id}/link-person", tags=["vehicles"])
async def link_vehicle_to_person(vehicle_id: str, person_id: str, user=Depends(require_auth)):
    db_execute("UPDATE vehicles SET owner_person_id = %s WHERE id = %s", (person_id, vehicle_id))
    db_execute(
        "UPDATE persons SET metadata = jsonb_set(metadata, '{linked_vehicle}', %s::jsonb) WHERE id = %s",
        (json.dumps(vehicle_id), person_id)
    )
    return {"status": "linked", "vehicle_id": vehicle_id, "person_id": person_id}


@app.get("/stats/system-health", tags=["system"])
async def system_health_detailed(user=Depends(require_admin)):
    db_ok = False
    redis_ok = False
    r2_ok = False
    models_ok = yolo_detector is not None and face_analyzer is not None
    try:
        db_execute("SELECT 1", fetch="scalar")
        db_ok = True
    except Exception:
        pass
    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:
        pass
    try:
        r2_client.list_buckets()
        r2_ok = True
    except Exception:
        pass
    overall = "healthy" if db_ok and redis_ok and models_ok else "degraded" if db_ok else "critical"
    return {
        "overall": overall,
        "database": db_ok,
        "redis": redis_ok,
        "r2_storage": r2_ok,
        "ai_models": models_ok,
        "active_sessions": len(active_sessions),
        "active_streams": len(active_ws_connections),
        "memory": {
            "known_faces": len(known_face_embeddings),
            "known_plates": len(known_plate_registry),
            "track_ids": len(track_position_histories),
        },
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.get("/stats/plates/top-seen", tags=["analytics"])
async def top_seen_plates(limit: int = 20, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT plate_number, seen_count, avg_speed_kph, max_speed_kph, flagged, stolen, last_seen FROM vehicles ORDER BY seen_count DESC LIMIT %s",
        (limit,), fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.post("/admin/broadcast-alert", tags=["admin"])
async def admin_broadcast_alert(
    alert_type: str,
    severity: str,
    description: str,
    user=Depends(require_admin)
):
    alert = AlertResult(
        alert_type=alert_type,
        severity=severity,
        description=description,
        metadata={"source": "admin_broadcast", "operator": user.get("sub", "admin")}
    )
    alert_id = db_save_alert(alert, "broadcast", None)
    for ws in list(active_ws_connections.values()):
        try:
            await ws.send_json({
                "event": "broadcast_alert",
                "alert_type": alert_type,
                "severity": severity,
                "description": description,
                "alert_id": alert_id,
                "ts": datetime.datetime.utcnow().isoformat(),
            })
        except Exception:
            pass
    return {"status": "broadcast", "alert_id": alert_id, "recipients": len(active_ws_connections)}


@app.get("/stats/top-alert-sessions", tags=["analytics"])
async def top_alert_sessions(limit: int = 10, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT session_id, COUNT(*) as alert_count, MAX(severity) as max_severity FROM alerts GROUP BY session_id ORDER BY alert_count DESC LIMIT %s",
        (limit,), fetch="all"
    )
    return [dict(r) for r in (rows or [])]


@app.get("/persons/{person_id}/associates", tags=["persons"])
async def get_person_associates(person_id: str, user=Depends(require_auth)):
    row = db_execute("SELECT known_associates FROM persons WHERE id = %s", (person_id,), fetch="one")
    if not row:
        raise HTTPException(status_code=404, detail="Person not found")
    associates_raw = row["known_associates"] or []
    if isinstance(associates_raw, str):
        associates_raw = json.loads(associates_raw)
    result = []
    for assoc in associates_raw:
        pid = assoc.get("id")
        if pid:
            p = db_execute("SELECT id, name, alias, threat_level, flagged, snapshot_url FROM persons WHERE id = %s", (pid,), fetch="one")
            if p:
                d = dict(p)
                d["relationship"] = assoc.get("relationship", "associate")
                result.append(d)
    return result


@app.get("/export/full-database", tags=["export"])
async def export_full_database_snapshot(user=Depends(require_admin)):
    persons = db_execute("SELECT id, name, age_estimate, gender, threat_level, flagged, seen_count FROM persons", fetch="all") or []
    vehicles = db_execute("SELECT id, plate_number, vehicle_type, flagged, stolen, seen_count FROM vehicles", fetch="all") or []
    alerts = db_execute("SELECT id, alert_type, severity, created_at, resolved FROM alerts ORDER BY created_at DESC LIMIT 10000", fetch="all") or []

    def safe_list(rows):
        result = []
        for r in rows:
            d = dict(r)
            for k, v in d.items():
                if isinstance(v, datetime.datetime):
                    d[k] = v.isoformat()
            result.append(d)
        return result

    snapshot = {
        "exported_at": datetime.datetime.utcnow().isoformat(),
        "system": "ARGUS",
        "version": "2.0.0",
        "persons": safe_list(persons),
        "vehicles": safe_list(vehicles),
        "alerts": safe_list(alerts),
        "total_persons": len(persons),
        "total_vehicles": len(vehicles),
        "total_alerts": len(alerts),
    }
    snap_json = json.dumps(snapshot, indent=2)
    return StreamingResponse(
        io.BytesIO(snap_json.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=argus_full_snapshot.json"}
    )



def compute_reentry_pattern(person_id: str) -> Dict:
    rows = db_execute(
        "SELECT detected_at, session_id FROM detections WHERE person_id = %s ORDER BY detected_at ASC",
        (person_id,), fetch="all"
    ) or []
    if len(rows) < 2:
        return {"reentries": 0, "intervals": [], "pattern": "single_visit"}
    intervals = []
    for i in range(1, len(rows)):
        a = rows[i-1]["detected_at"]
        b = rows[i]["detected_at"]
        if isinstance(a, datetime.datetime) and isinstance(b, datetime.datetime):
            diff = (b - a).total_seconds()
            intervals.append(round(diff, 1))
    avg_interval = round(float(np.mean(intervals)), 1) if intervals else 0
    pattern = "frequent" if avg_interval < 300 else "periodic" if avg_interval < 3600 else "occasional"
    return {
        "reentries": len(rows) - 1,
        "avg_interval_seconds": avg_interval,
        "intervals": intervals[:20],
        "pattern": pattern,
    }


def compute_session_activity_curve(session_id: str) -> List[Dict]:
    rows = db_execute(
        """SELECT DATE_TRUNC('minute', detected_at) as minute, COUNT(*) as count
           FROM detections WHERE session_id = %s
           GROUP BY minute ORDER BY minute ASC""",
        (session_id,), fetch="all"
    ) or []
    result = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("minute"), datetime.datetime):
            d["minute"] = d["minute"].isoformat()
        result.append(d)
    return result


def detect_repetitive_plate(plate: str, window_minutes: int = 10) -> int:
    count = db_execute(
        "SELECT COUNT(*) FROM plate_reads WHERE plate_number = %s AND read_at > NOW() - INTERVAL '%s minutes'",
        (plate, window_minutes), fetch="scalar"
    )
    return int(count or 0)


def flag_high_frequency_plate(plate: str, threshold: int = 5, window_minutes: int = 10) -> bool:
    count = detect_repetitive_plate(plate, window_minutes)
    return count >= threshold


def compute_session_coverage(session_id: str) -> Dict:
    total = db_execute("SELECT COUNT(*) FROM detections WHERE session_id = %s", (session_id,), fetch="scalar") or 0
    person_det = db_execute("SELECT COUNT(*) FROM detections WHERE session_id = %s AND detection_type = 'person'", (session_id,), fetch="scalar") or 0
    vehicle_det = db_execute("SELECT COUNT(*) FROM detections WHERE session_id = %s AND detection_type = 'vehicle'", (session_id,), fetch="scalar") or 0
    pose_det = db_execute("SELECT COUNT(*) FROM detections WHERE session_id = %s AND pose_threat = TRUE", (session_id,), fetch="scalar") or 0
    returning = db_execute("SELECT COUNT(*) FROM detections WHERE session_id = %s AND is_returning = TRUE", (session_id,), fetch="scalar") or 0
    flagged_det = db_execute("SELECT COUNT(*) FROM detections WHERE session_id = %s AND is_flagged = TRUE", (session_id,), fetch="scalar") or 0
    return {
        "session_id": session_id,
        "total_detections": int(total),
        "person_detections": int(person_det),
        "vehicle_detections": int(vehicle_det),
        "pose_threat_detections": int(pose_det),
        "returning_detections": int(returning),
        "flagged_detections": int(flagged_det),
        "person_pct": round(int(person_det) / max(int(total), 1) * 100, 1),
        "vehicle_pct": round(int(vehicle_det) / max(int(total), 1) * 100, 1),
    }


def rank_persons_by_suspicion(session_id: Optional[str] = None) -> List[Dict]:
    if session_id:
        persons = db_execute(
            "SELECT DISTINCT p.id, p.name, p.threat_score, p.flagged, p.threat_level, p.seen_count FROM persons p JOIN detections d ON d.person_id = p.id WHERE d.session_id = %s ORDER BY p.threat_score DESC LIMIT 20",
            (session_id,), fetch="all"
        )
    else:
        persons = db_execute(
            "SELECT id, name, threat_score, flagged, threat_level, seen_count FROM persons ORDER BY threat_score DESC LIMIT 20",
            fetch="all"
        )
    result = []
    for r in (persons or []):
        d = dict(r)
        result.append(d)
    return result


def detect_unusual_time_activity(session_id: str) -> List[Dict]:
    rows = db_execute(
        """SELECT EXTRACT(HOUR FROM detected_at) as hour, COUNT(*) as count
           FROM detections WHERE session_id = %s
           GROUP BY hour ORDER BY count DESC""",
        (session_id,), fetch="all"
    ) or []
    unusual = []
    for r in rows:
        hour = int(r["hour"])
        count = int(r["count"])
        if 0 <= hour < 5 and count > 0:
            unusual.append({"hour": hour, "count": count, "flag": "late_night_activity"})
    return unusual


def build_incident_timeline(session_id: str) -> List[Dict]:
    alerts = db_execute(
        "SELECT id, alert_type, severity, created_at, description, person_id, vehicle_id FROM alerts WHERE session_id = %s ORDER BY created_at ASC",
        (session_id,), fetch="all"
    ) or []
    behaviors = db_execute(
        "SELECT id, behavior_type, detected_at, person_id, track_id, duration_seconds FROM behavior_events WHERE session_id = %s ORDER BY detected_at ASC",
        (session_id,), fetch="all"
    ) or []
    audio_ev = db_execute(
        "SELECT id, event_type, detected_at, confidence FROM audio_events WHERE session_id = %s ORDER BY detected_at ASC",
        (session_id,), fetch="all"
    ) or []
    timeline = []
    for r in alerts:
        d = dict(r)
        d["timeline_type"] = "alert"
        d["ts"] = d["created_at"].isoformat() if isinstance(d.get("created_at"), datetime.datetime) else str(d.get("created_at"))
        timeline.append(d)
    for r in behaviors:
        d = dict(r)
        d["timeline_type"] = "behavior"
        d["ts"] = d["detected_at"].isoformat() if isinstance(d.get("detected_at"), datetime.datetime) else str(d.get("detected_at"))
        timeline.append(d)
    for r in audio_ev:
        d = dict(r)
        d["timeline_type"] = "audio"
        d["ts"] = d["detected_at"].isoformat() if isinstance(d.get("detected_at"), datetime.datetime) else str(d.get("detected_at"))
        timeline.append(d)
    timeline.sort(key=lambda x: x.get("ts", ""))
    return timeline


@app.get("/persons/{person_id}/reentry-pattern", tags=["persons"])
async def person_reentry_pattern(person_id: str, user=Depends(require_auth)):
    return compute_reentry_pattern(person_id)


@app.get("/sessions/{session_id}/activity-curve", tags=["sessions"])
async def session_activity_curve(session_id: str, user=Depends(require_auth)):
    return {"session_id": session_id, "curve": compute_session_activity_curve(session_id)}


@app.get("/sessions/{session_id}/incident-timeline", tags=["sessions"])
async def session_incident_timeline(session_id: str, user=Depends(require_auth)):
    return {"session_id": session_id, "timeline": build_incident_timeline(session_id)}


@app.get("/sessions/{session_id}/coverage", tags=["sessions"])
async def session_coverage(session_id: str, user=Depends(require_auth)):
    return compute_session_coverage(session_id)


@app.get("/sessions/{session_id}/unusual-activity", tags=["sessions"])
async def session_unusual_activity(session_id: str, user=Depends(require_auth)):
    return {"session_id": session_id, "unusual_hours": detect_unusual_time_activity(session_id)}


@app.get("/persons/ranked/suspicion", tags=["analytics"])
async def ranked_suspicious_persons(session_id: Optional[str] = None, user=Depends(require_auth)):
    return rank_persons_by_suspicion(session_id)


@app.get("/vehicles/plates/high-frequency", tags=["vehicles"])
async def high_frequency_plates(window_minutes: int = 10, threshold: int = 3, user=Depends(require_auth)):
    rows = db_execute(
        """SELECT plate_number, COUNT(*) as reads FROM plate_reads
           WHERE read_at > NOW() - INTERVAL '%s minutes'
           GROUP BY plate_number HAVING COUNT(*) >= %s
           ORDER BY reads DESC""",
        (window_minutes, threshold), fetch="all"
    )
    return [dict(r) for r in (rows or [])]


@app.get("/analyze/persons/clustering", tags=["analysis"])
async def person_clustering_stats(session_id: Optional[str] = None, user=Depends(require_auth)):
    if session_id and session_id in crowd_density_grid:
        grid = crowd_density_grid[session_id]
        return {
            "session_id": session_id,
            "hotspots": compute_hotspot_zones(session_id),
            "total_density": int(np.sum(grid)),
            "peak_cell": int(np.max(grid)),
        }
    return {"message": "No active crowd data for this session"}


@app.get("/analyze/audio/threat-probability", tags=["audio"])
async def audio_threat_probability(session_id: Optional[str] = None, user=Depends(require_auth)):
    if session_id:
        events = db_execute(
            "SELECT event_type, AVG(confidence) as avg_conf, COUNT(*) as c FROM audio_events WHERE session_id = %s GROUP BY event_type",
            (session_id,), fetch="all"
        )
    else:
        events = db_execute(
            "SELECT event_type, AVG(confidence) as avg_conf, COUNT(*) as c FROM audio_events GROUP BY event_type",
            fetch="all"
        )
    return [dict(r) for r in (events or [])]


@app.get("/stats/daily-summary", tags=["analytics"])
async def daily_summary(user=Depends(require_auth)):
    today = datetime.date.today()
    start = datetime.datetime.combine(today, datetime.time.min)
    end = datetime.datetime.combine(today, datetime.time.max)
    persons_today = db_execute("SELECT COUNT(*) FROM persons WHERE first_seen BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    vehicles_today = db_execute("SELECT COUNT(*) FROM vehicles WHERE first_seen BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    alerts_today = db_execute("SELECT COUNT(*) FROM alerts WHERE created_at BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    critical_today = db_execute("SELECT COUNT(*) FROM alerts WHERE created_at BETWEEN %s AND %s AND severity = 'critical'", (start, end), fetch="scalar") or 0
    detections_today = db_execute("SELECT COUNT(*) FROM detections WHERE detected_at BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    sessions_today = db_execute("SELECT COUNT(*) FROM sessions WHERE created_at BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    plate_reads_today = db_execute("SELECT COUNT(*) FROM plate_reads WHERE read_at BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    speed_violations_today = db_execute("SELECT COUNT(*) FROM plate_reads WHERE read_at BETWEEN %s AND %s AND speed_at_read > 100", (start, end), fetch="scalar") or 0
    return {
        "date": today.isoformat(),
        "new_persons": int(persons_today),
        "new_vehicles": int(vehicles_today),
        "total_alerts": int(alerts_today),
        "critical_alerts": int(critical_today),
        "total_detections": int(detections_today),
        "sessions_run": int(sessions_today),
        "plate_reads": int(plate_reads_today),
        "speed_violations": int(speed_violations_today),
    }


@app.get("/stats/weekly-summary", tags=["analytics"])
async def weekly_summary(user=Depends(require_auth)):
    start = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    rows = db_execute(
        """SELECT
            DATE_TRUNC('day', created_at)::date as day,
            COUNT(*) as total_alerts,
            SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical,
            SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high,
            SUM(CASE WHEN severity = 'medium' THEN 1 ELSE 0 END) as medium
           FROM alerts WHERE created_at >= %s
           GROUP BY day ORDER BY day ASC""",
        (start,), fetch="all"
    ) or []
    result = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("day"), (datetime.date, datetime.datetime)):
            d["day"] = d["day"].isoformat()
        result.append(d)
    return result



def compute_detection_confidence_stats(session_id: str) -> Dict:
    rows = db_execute(
        "SELECT detection_type, AVG(confidence) as avg_conf, MIN(confidence) as min_conf, MAX(confidence) as max_conf, COUNT(*) as total FROM detections WHERE session_id = %s GROUP BY detection_type",
        (session_id,), fetch="all"
    ) or []
    return {r["detection_type"]: {k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in dict(r).items() if k != "detection_type"} for r in rows}


def build_forensic_report(person_id: str, vehicle_id: Optional[str] = None) -> Dict:
    dossier = build_full_person_dossier(person_id)
    v_dossier = build_full_vehicle_dossier(vehicle_id) if vehicle_id else {}
    reentry = compute_reentry_pattern(person_id)
    cross = track_cross_session_person(person_id)
    risk = db_execute("SELECT threat_score, threat_level, flag_reason FROM persons WHERE id = %s", (person_id,), fetch="one")
    risk_data = dict(risk) if risk else {}
    return {
        "report_type": "forensic",
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "person": dossier,
        "vehicle": v_dossier,
        "reentry_pattern": reentry,
        "cross_session_activity": cross,
        "risk_assessment": risk_data,
        "system": "ARGUS v2.0.0",
    }


def auto_escalate_alert(alert_id: str, current_level: int) -> bool:
    if current_level >= 3:
        return False
    db_execute(
        "UPDATE alerts SET escalated = TRUE, escalation_level = %s, updated_at = NOW() WHERE id = %s",
        (current_level + 1, alert_id)
    )
    return True


async def run_auto_escalation_job():
    while True:
        await asyncio.sleep(120)
        try:
            stale_criticals = db_execute(
                "SELECT id, escalation_level FROM alerts WHERE severity = 'critical' AND resolved = FALSE AND acknowledged = FALSE AND created_at < NOW() - INTERVAL '5 minutes'",
                fetch="all"
            ) or []
            for row in stale_criticals:
                auto_escalate_alert(str(row["id"]), int(row["escalation_level"]))
        except Exception as e:
            logger.error(f"Auto-escalation error: {e}")


@app.on_event("startup")
async def start_escalation_job():
    asyncio.create_task(run_auto_escalation_job())


def compute_motion_zones(session_id: str, frame: np.ndarray) -> List[Dict]:
    bg_sub = motion_background_subtractors.get(session_id)
    if bg_sub is None:
        return []
    mask = bg_sub.apply(frame, learningRate=0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        zones.append({"x": x, "y": y, "w": w, "h": h, "area": int(area)})
    zones.sort(key=lambda z: z["area"], reverse=True)
    return zones[:10]


def smooth_track_positions(track_id: int, window: int = 5) -> List[List[float]]:
    positions = track_position_histories.get(track_id, [])
    if len(positions) < window:
        return positions
    smoothed = []
    for i in range(len(positions)):
        start = max(0, i - window // 2)
        end = min(len(positions), i + window // 2 + 1)
        chunk = positions[start:end]
        avg_x = sum(p[0] for p in chunk) / len(chunk)
        avg_y = sum(p[1] for p in chunk) / len(chunk)
        smoothed.append([round(avg_x, 2), round(avg_y, 2)])
    return smoothed


def compute_person_speed_estimate(track_id: int, fps: float) -> float:
    positions = track_position_histories.get(track_id, [])
    if len(positions) < 2:
        return 0.0
    dx = positions[-1][0] - positions[-2][0]
    dy = positions[-1][1] - positions[-2][1]
    pixel_dist = math.sqrt(dx**2 + dy**2)
    pixels_per_meter = 50.0
    speed_ms = (pixel_dist / pixels_per_meter) * fps
    return round(speed_ms * 3.6, 2)


def build_plate_encounter_map(plate: str) -> List[Dict]:
    rows = db_execute(
        "SELECT session_id, read_at, speed_at_read, snapshot_url FROM plate_reads WHERE plate_number = %s ORDER BY read_at DESC LIMIT 50",
        (plate,), fetch="all"
    ) or []
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


def compute_face_recognition_accuracy(session_id: str) -> Dict:
    total = db_execute("SELECT COUNT(*) FROM detections WHERE session_id = %s AND detection_type = 'person'", (session_id,), fetch="scalar") or 0
    with_face = db_execute("SELECT COUNT(*) FROM detections WHERE session_id = %s AND person_id IS NOT NULL", (session_id,), fetch="scalar") or 0
    returning = db_execute("SELECT COUNT(*) FROM detections WHERE session_id = %s AND is_returning = TRUE", (session_id,), fetch="scalar") or 0
    return {
        "total_person_detections": int(total),
        "face_matched": int(with_face),
        "returning_recognized": int(returning),
        "recognition_rate_pct": round(int(with_face) / max(int(total), 1) * 100, 2),
    }


def generate_operator_summary(operator: str) -> Dict:
    sessions = db_execute("SELECT COUNT(*) FROM sessions WHERE operator = %s", (operator,), fetch="scalar") or 0
    flags = db_execute("SELECT COUNT(*) FROM persons WHERE flag_operator = %s", (operator,), fetch="scalar") or 0
    resolutions = db_execute("SELECT COUNT(*) FROM alerts WHERE resolved_by = %s", (operator,), fetch="scalar") or 0
    acknowledgements = db_execute("SELECT COUNT(*) FROM alerts WHERE acknowledged_by = %s", (operator,), fetch="scalar") or 0
    return {
        "operator": operator,
        "total_sessions": int(sessions),
        "persons_flagged": int(flags),
        "alerts_resolved": int(resolutions),
        "alerts_acknowledged": int(acknowledgements),
    }


def find_colocated_persons(session_id: str, time_window_seconds: float = 30.0) -> List[Dict]:
    rows = db_execute(
        "SELECT person_id, detected_at FROM detections WHERE session_id = %s AND person_id IS NOT NULL ORDER BY detected_at ASC",
        (session_id,), fetch="all"
    ) or []
    groups = []
    processed = set()
    for i, ri in enumerate(rows):
        if i in processed:
            continue
        group = [str(ri["person_id"])]
        ti = ri["detected_at"]
        for j, rj in enumerate(rows[i+1:], start=i+1):
            if j in processed:
                continue
            tj = rj["detected_at"]
            diff = abs((tj - ti).total_seconds()) if isinstance(ti, datetime.datetime) and isinstance(tj, datetime.datetime) else 999
            if diff <= time_window_seconds and str(rj["person_id"]) not in group:
                group.append(str(rj["person_id"]))
                processed.add(j)
        if len(group) >= 2:
            groups.append({"persons": group, "size": len(group), "base_time": ti.isoformat() if isinstance(ti, datetime.datetime) else str(ti)})
        processed.add(i)
    return groups


@app.get("/sessions/{session_id}/confidence-stats", tags=["sessions"])
async def session_confidence_stats(session_id: str, user=Depends(require_auth)):
    return compute_detection_confidence_stats(session_id)


@app.get("/sessions/{session_id}/face-recognition-accuracy", tags=["sessions"])
async def face_recognition_accuracy(session_id: str, user=Depends(require_auth)):
    return compute_face_recognition_accuracy(session_id)


@app.get("/sessions/{session_id}/colocated-persons", tags=["sessions"])
async def colocated_persons(session_id: str, window: float = 30.0, user=Depends(require_auth)):
    return {"session_id": session_id, "groups": find_colocated_persons(session_id, window)}


@app.post("/analyze/forensic", tags=["analysis"])
async def forensic_report(person_id: str, vehicle_id: Optional[str] = None, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(executor, build_forensic_report, person_id, vehicle_id)
    return report


@app.get("/vehicles/plates/{plate}/encounters", tags=["vehicles"])
async def plate_encounter_map(plate: str, user=Depends(require_auth)):
    return {"plate": plate, "encounters": build_plate_encounter_map(plate)}


@app.get("/analyze/tracks/smooth/{track_id}", tags=["analysis"])
async def smooth_track(track_id: int, user=Depends(require_auth)):
    smoothed = smooth_track_positions(track_id)
    raw = track_position_histories.get(track_id, [])
    future = compute_trajectory_prediction(track_id)
    return {
        "track_id": track_id,
        "raw_positions": raw,
        "smoothed_positions": smoothed,
        "predicted_path": future,
        "direction": estimate_person_direction(track_id),
        "speed_estimate_kph": compute_person_speed_estimate(track_id, 30.0),
    }


@app.get("/operators/{username}/summary", tags=["auth"])
async def operator_summary(username: str, user=Depends(require_admin)):
    return generate_operator_summary(username)


@app.get("/export/forensic/{person_id}", tags=["export"])
async def export_forensic_report(person_id: str, vehicle_id: Optional[str] = None, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(executor, build_forensic_report, person_id, vehicle_id)
    report_json = json.dumps(report, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(report_json.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=argus_forensic_{person_id[:8]}.json"}
    )


@app.get("/stats/faces/quality-distribution", tags=["analytics"])
async def face_quality_distribution(user=Depends(require_auth)):
    rows = db_execute(
        """SELECT
            SUM(CASE WHEN face_quality_score >= 80 THEN 1 ELSE 0 END) as high_quality,
            SUM(CASE WHEN face_quality_score >= 50 AND face_quality_score < 80 THEN 1 ELSE 0 END) as medium_quality,
            SUM(CASE WHEN face_quality_score < 50 THEN 1 ELSE 0 END) as low_quality,
            AVG(face_quality_score) as avg_quality
           FROM persons WHERE embedding IS NOT NULL""",
        fetch="one"
    )
    return dict(rows) if rows else {}


@app.get("/stats/alerts/response-time", tags=["analytics"])
async def alert_response_time(user=Depends(require_auth)):
    rows = db_execute(
        """SELECT severity,
            AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))) as avg_response_seconds,
            MIN(EXTRACT(EPOCH FROM (resolved_at - created_at))) as min_response_seconds,
            MAX(EXTRACT(EPOCH FROM (resolved_at - created_at))) as max_response_seconds,
            COUNT(*) as total
           FROM alerts WHERE resolved = TRUE AND resolved_at IS NOT NULL
           GROUP BY severity ORDER BY severity""",
        fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if v is not None and not isinstance(v, (str, bool)):
                try:
                    d[k] = round(float(v), 2)
                except Exception:
                    pass
        result.append(d)
    return result


@app.get("/stats/sessions/longest", tags=["analytics"])
async def longest_sessions(limit: int = 10, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, operator, source_type, duration_seconds, total_alerts, total_persons_unique, created_at FROM sessions WHERE duration_seconds IS NOT NULL ORDER BY duration_seconds DESC LIMIT %s",
        (limit,), fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.get("/stats/persons/most-emotional", tags=["analytics"])
async def most_emotional_persons(emotion: str = "angry", limit: int = 20, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT id, name, dominant_emotion, seen_count, flagged, snapshot_url FROM persons WHERE dominant_emotion = %s ORDER BY seen_count DESC LIMIT %s",
        (emotion, limit), fetch="all"
    )
    return [dict(r) for r in (rows or [])]


@app.get("/live/speed-violations", tags=["live"])
async def live_speed_violations(threshold: float = 100.0, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT plate_number, speed_at_read, read_at, session_id, snapshot_url FROM plate_reads WHERE speed_at_read > %s AND read_at > NOW() - INTERVAL '1 hour' ORDER BY read_at DESC LIMIT 30",
        (threshold,), fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.get("/live/behavior-feed", tags=["live"])
async def live_behavior_feed(limit: int = 50, user=Depends(require_auth)):
    rows = db_execute(
        "SELECT * FROM behavior_events WHERE detected_at > NOW() - INTERVAL '30 minutes' ORDER BY detected_at DESC LIMIT %s",
        (limit,), fetch="all"
    )
    result = []
    for r in (rows or []):
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.websocket("/ws/speed-alerts")
async def websocket_speed_alerts(websocket: WebSocket):
    await websocket.accept()
    try:
        last_ts = datetime.datetime.utcnow() - datetime.timedelta(seconds=5)
        while True:
            await asyncio.sleep(1)
            rows = db_execute(
                "SELECT plate_number, speed_at_read, read_at, session_id FROM plate_reads WHERE speed_at_read > 100 AND read_at > %s ORDER BY read_at DESC",
                (last_ts,), fetch="all"
            ) or []
            if rows:
                last_ts = datetime.datetime.utcnow()
                for r in rows:
                    d = dict(r)
                    for k, v in d.items():
                        if isinstance(v, datetime.datetime):
                            d[k] = v.isoformat()
                    await websocket.send_json({"event": "speed_violation", "data": d})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Speed alerts WS error: {e}")


@app.websocket("/ws/behavior-feed")
async def websocket_behavior_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        last_ts = datetime.datetime.utcnow() - datetime.timedelta(seconds=5)
        while True:
            await asyncio.sleep(2)
            rows = db_execute(
                "SELECT * FROM behavior_events WHERE detected_at > %s ORDER BY detected_at DESC LIMIT 10",
                (last_ts,), fetch="all"
            ) or []
            if rows:
                last_ts = datetime.datetime.utcnow()
                payload = []
                for r in rows:
                    d = dict(r)
                    for k, v in d.items():
                        if isinstance(v, datetime.datetime):
                            d[k] = v.isoformat()
                    payload.append(d)
                await websocket.send_json({"event": "behavior_update", "events": payload})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Behavior feed WS error: {e}")


@app.get("/system/endpoints-count", tags=["system"])
async def endpoints_count():
    rest = sum(1 for r in app.routes if hasattr(r, "methods"))
    ws = sum(1 for r in app.routes if not hasattr(r, "methods") and hasattr(r, "path"))
    return {"rest_endpoints": rest, "websocket_endpoints": ws, "total": rest + ws}


@app.get("/system/version", tags=["system"])
async def system_version():
    return {
        "name": "ARGUS",
        "version": "2.0.0",
        "build": "production",
        "python": "3.11+",
        "framework": "FastAPI",
        "ai_stack": ["YOLOv8x", "InsightFace buffalo_l", "DeepFace", "PaddleOCR"],
        "database": "PostgreSQL + pgvector",
        "cache": "Redis (Upstash)",
        "storage": "Cloudflare R2",
        "architecture": "single-file monolith",
        "author": "ARGUS System",
    }



def compute_alert_density_per_hour(session_id: str) -> List[Dict]:
    rows = db_execute(
        """SELECT DATE_TRUNC('hour', created_at) as hour, severity, COUNT(*) as count
           FROM alerts WHERE session_id = %s
           GROUP BY hour, severity ORDER BY hour ASC""",
        (session_id,), fetch="all"
    ) or []
    result = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("hour"), datetime.datetime):
            d["hour"] = d["hour"].isoformat()
        result.append(d)
    return result


def compute_plate_confidence_distribution(session_id: Optional[str] = None) -> Dict:
    if session_id:
        rows = db_execute(
            "SELECT confidence FROM plate_reads WHERE session_id = %s AND confidence > 0",
            (session_id,), fetch="all"
        ) or []
    else:
        rows = db_execute("SELECT confidence FROM plate_reads WHERE confidence > 0 LIMIT 5000", fetch="all") or []
    confs = [float(r["confidence"]) for r in rows]
    if not confs:
        return {}
    arr = np.array(confs)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "sample_count": len(confs),
    }


def compute_session_peak_activity(session_id: str) -> Dict:
    rows = db_execute(
        """SELECT DATE_TRUNC('minute', detected_at) as minute, COUNT(*) as count
           FROM detections WHERE session_id = %s
           GROUP BY minute ORDER BY count DESC LIMIT 1""",
        (session_id,), fetch="one"
    )
    if not rows:
        return {}
    d = dict(rows)
    if isinstance(d.get("minute"), datetime.datetime):
        d["minute"] = d["minute"].isoformat()
    return d


def check_person_on_watchlists(person_id: str) -> List[Dict]:
    rows = db_execute(
        "SELECT list_name, list_type, reason, priority, added_at, added_by FROM watchlist WHERE person_id = %s AND active = TRUE ORDER BY priority DESC",
        (person_id,), fetch="all"
    ) or []
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


def check_plate_on_watchlists(plate: str) -> List[Dict]:
    rows = db_execute(
        "SELECT list_name, list_type, reason, priority, added_at, added_by FROM watchlist WHERE vehicle_plate = %s AND active = TRUE ORDER BY priority DESC",
        (plate,), fetch="all"
    ) or []
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


def run_end_of_session_analysis(session_id: str) -> Dict:
    coverage = compute_session_coverage(session_id)
    peak = compute_session_peak_activity(session_id)
    density = compute_alert_density_per_hour(session_id)
    unusual = detect_unusual_time_activity(session_id)
    demo = build_demographic_report(session_id)
    zone_stats = compute_zone_statistics(session_id)
    conf_stats = compute_detection_confidence_stats(session_id)
    face_acc = compute_face_recognition_accuracy(session_id)
    incident_tl = build_incident_timeline(session_id)
    return {
        "session_id": session_id,
        "analysis_generated_at": datetime.datetime.utcnow().isoformat(),
        "coverage": coverage,
        "peak_activity": peak,
        "alert_density_by_hour": density,
        "unusual_hour_activity": unusual,
        "demographics": demo,
        "zone_statistics": zone_stats,
        "confidence_statistics": conf_stats,
        "face_recognition_accuracy": face_acc,
        "incident_timeline": incident_tl,
    }


def generate_argus_daily_digest() -> Dict:
    today = datetime.date.today()
    start = datetime.datetime.combine(today, datetime.time.min)
    end = datetime.datetime.combine(today, datetime.time.max)
    new_persons = db_execute("SELECT COUNT(*) FROM persons WHERE first_seen BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    new_vehicles = db_execute("SELECT COUNT(*) FROM vehicles WHERE first_seen BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    total_alerts = db_execute("SELECT COUNT(*) FROM alerts WHERE created_at BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    critical_alerts = db_execute("SELECT COUNT(*) FROM alerts WHERE created_at BETWEEN %s AND %s AND severity = 'critical'", (start, end), fetch="scalar") or 0
    flagged_today = db_execute("SELECT COUNT(*) FROM persons WHERE flag_timestamp BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    top_alert_type = db_execute(
        "SELECT alert_type, COUNT(*) as c FROM alerts WHERE created_at BETWEEN %s AND %s GROUP BY alert_type ORDER BY c DESC LIMIT 1",
        (start, end), fetch="one"
    )
    top_type = dict(top_alert_type) if top_alert_type else {}
    speed_violations = db_execute("SELECT COUNT(*) FROM plate_reads WHERE read_at BETWEEN %s AND %s AND speed_at_read > 100", (start, end), fetch="scalar") or 0
    audio_threats = db_execute("SELECT COUNT(*) FROM audio_events WHERE detected_at BETWEEN %s AND %s", (start, end), fetch="scalar") or 0
    return {
        "digest_date": today.isoformat(),
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "new_persons_detected": int(new_persons),
        "new_vehicles_detected": int(new_vehicles),
        "total_alerts": int(total_alerts),
        "critical_alerts": int(critical_alerts),
        "persons_flagged_today": int(flagged_today),
        "speed_violations": int(speed_violations),
        "audio_threats_detected": int(audio_threats),
        "top_alert_type": top_type,
        "system": "ARGUS v2.0.0",
    }


def get_active_session_summary() -> List[Dict]:
    result = []
    for sid, s in active_sessions.items():
        up = round(time.time() - s.get("started_at", time.time()), 1)
        result.append({
            "session_id": sid,
            "source_type": s.get("source_type"),
            "uptime_seconds": up,
            "fps": s.get("fps", 0),
            "frame_count": s.get("frame_count", 0),
            "alert_count": s.get("alert_count", 0),
            "unique_persons": len(s.get("unique_persons", set())),
            "unique_vehicles": len(s.get("unique_vehicles", set())),
            "night_mode": night_mode_active.get(sid, False),
            "ws_connected": sid in active_ws_connections,
            "audio_connected": sid in audio_ws_connections,
            "operator": s.get("operator", "system"),
        })
    return result


@app.get("/sessions/{session_id}/end-analysis", tags=["sessions"])
async def session_end_analysis(session_id: str, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    analysis = await loop.run_in_executor(executor, run_end_of_session_analysis, session_id)
    return analysis


@app.get("/sessions/{session_id}/alert-density", tags=["sessions"])
async def session_alert_density(session_id: str, user=Depends(require_auth)):
    return {"session_id": session_id, "density": compute_alert_density_per_hour(session_id)}


@app.get("/sessions/{session_id}/peak-activity", tags=["sessions"])
async def session_peak_activity(session_id: str, user=Depends(require_auth)):
    return compute_session_peak_activity(session_id)


@app.get("/stats/daily-digest", tags=["analytics"])
async def daily_digest(user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_argus_daily_digest)


@app.get("/live/active-summary", tags=["live"])
async def active_session_summary(user=Depends(require_auth)):
    return {"sessions": get_active_session_summary(), "count": len(active_sessions)}


@app.get("/persons/{person_id}/watchlist-check", tags=["persons"])
async def person_watchlist_check(person_id: str, user=Depends(require_auth)):
    entries = check_person_on_watchlists(person_id)
    return {"person_id": person_id, "on_watchlists": entries, "watchlist_count": len(entries)}


@app.get("/vehicles/plates/{plate}/watchlist-check", tags=["vehicles"])
async def plate_watchlist_check(plate: str, user=Depends(require_auth)):
    entries = check_plate_on_watchlists(plate)
    return {"plate": plate, "on_watchlists": entries, "watchlist_count": len(entries)}


@app.get("/stats/plate-confidence", tags=["analytics"])
async def plate_confidence_stats(session_id: Optional[str] = None, user=Depends(require_auth)):
    return compute_plate_confidence_distribution(session_id)


@app.get("/export/end-analysis/{session_id}", tags=["export"])
async def export_end_analysis(session_id: str, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    analysis = await loop.run_in_executor(executor, run_end_of_session_analysis, session_id)
    out = json.dumps(analysis, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(out.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=argus_analysis_{session_id[:8]}.json"}
    )


@app.get("/export/daily-digest", tags=["export"])
async def export_daily_digest(user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    digest = await loop.run_in_executor(executor, generate_argus_daily_digest)
    out = json.dumps(digest, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(out.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=argus_digest_{datetime.date.today().isoformat()}.json"}
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse({"detail": "Not found", "path": str(request.url.path)}, status_code=404)


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    return JSONResponse({"detail": "Rate limit exceeded. Slow down."}, status_code=429)



def compute_person_dwell_time(person_id: str, session_id: str) -> Dict:
    rows = db_execute(
        "SELECT detected_at FROM detections WHERE person_id = %s AND session_id = %s ORDER BY detected_at ASC",
        (person_id, session_id), fetch="all"
    ) or []
    if len(rows) < 2:
        return {"person_id": person_id, "dwell_seconds": 0, "first_seen": None, "last_seen": None}
    first = rows[0]["detected_at"]
    last = rows[-1]["detected_at"]
    dwell = (last - first).total_seconds() if isinstance(first, datetime.datetime) and isinstance(last, datetime.datetime) else 0
    return {
        "person_id": person_id,
        "session_id": session_id,
        "dwell_seconds": round(dwell, 2),
        "dwell_minutes": round(dwell / 60, 2),
        "first_seen": first.isoformat() if isinstance(first, datetime.datetime) else str(first),
        "last_seen": last.isoformat() if isinstance(last, datetime.datetime) else str(last),
        "total_detections": len(rows),
    }


def rank_sessions_by_threat(limit: int = 10) -> List[Dict]:
    rows = db_execute(
        """SELECT s.id, s.operator, s.source_type, s.created_at,
           COUNT(a.id) as total_alerts,
           SUM(CASE WHEN a.severity = 'critical' THEN 1 ELSE 0 END) as critical_count,
           SUM(CASE WHEN a.severity = 'high' THEN 1 ELSE 0 END) as high_count
           FROM sessions s LEFT JOIN alerts a ON a.session_id = s.id
           GROUP BY s.id, s.operator, s.source_type, s.created_at
           ORDER BY critical_count DESC, high_count DESC LIMIT %s""",
        (limit,), fetch="all"
    ) or []
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
            elif v is None:
                d[k] = 0
        result.append(d)
    return result


def compute_camera_coverage_score(session_id: str, frame_w: int = 1280, frame_h: int = 720) -> Dict:
    if session_id not in attention_heatmap:
        return {"score": 0, "covered_pct": 0}
    hm = attention_heatmap[session_id]
    total_cells = hm.size
    covered_cells = int(np.sum(hm > 0))
    coverage_pct = round(covered_cells / max(total_cells, 1) * 100, 2)
    score = min(100.0, coverage_pct * 1.5)
    hot_cells = int(np.sum(hm > np.mean(hm) + np.std(hm)))
    return {
        "session_id": session_id,
        "coverage_score": round(score, 2),
        "covered_cells_pct": coverage_pct,
        "total_grid_cells": total_cells,
        "covered_cells": covered_cells,
        "hotspot_cells": hot_cells,
    }


def detect_pattern_of_life(person_id: str) -> Dict:
    rows = db_execute(
        "SELECT detected_at FROM detections WHERE person_id = %s ORDER BY detected_at ASC",
        (person_id,), fetch="all"
    ) or []
    if not rows:
        return {"person_id": person_id, "pattern": "unknown", "data_points": 0}
    hours = [r["detected_at"].hour for r in rows if isinstance(r["detected_at"], datetime.datetime)]
    weekdays = [r["detected_at"].weekday() for r in rows if isinstance(r["detected_at"], datetime.datetime)]
    if not hours:
        return {"person_id": person_id, "pattern": "unknown", "data_points": len(rows)}
    hour_counter = Counter(hours)
    weekday_counter = Counter(weekdays)
    peak_hour = hour_counter.most_common(1)[0][0] if hour_counter else None
    peak_day = weekday_counter.most_common(1)[0][0] if weekday_counter else None
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    routine = "unknown"
    if peak_hour is not None:
        if 6 <= peak_hour <= 9:
            routine = "morning_commuter"
        elif 10 <= peak_hour <= 15:
            routine = "daytime_active"
        elif 16 <= peak_hour <= 19:
            routine = "evening_commuter"
        elif 20 <= peak_hour <= 23:
            routine = "night_active"
        elif 0 <= peak_hour <= 5:
            routine = "nocturnal"
    return {
        "person_id": person_id,
        "pattern": routine,
        "peak_hour": peak_hour,
        "peak_day": day_names[peak_day] if peak_day is not None else None,
        "hour_distribution": dict(hour_counter),
        "weekday_distribution": {day_names[k]: v for k, v in weekday_counter.items()},
        "data_points": len(rows),
    }


def build_zone_report(session_id: str) -> Dict:
    intrusions = zone_intrusion_log.get(session_id, [])
    if not intrusions:
        return {"session_id": session_id, "total_intrusions": 0, "zones": {}}
    zone_groups: Dict[str, List] = defaultdict(list)
    for entry in intrusions:
        zone_key = str(entry.get("zone", "unknown"))
        zone_groups[zone_key].append(entry)
    zones_summary = {}
    for zk, entries in zone_groups.items():
        persons = list(set(e.get("person_id") for e in entries if e.get("person_id")))
        tracks = list(set(e.get("track_id") for e in entries if e.get("track_id")))
        zones_summary[zk] = {
            "total_breaches": len(entries),
            "unique_persons": len(persons),
            "unique_tracks": len(tracks),
            "person_ids": persons,
            "first_breach": entries[0].get("ts"),
            "last_breach": entries[-1].get("ts"),
        }
    return {
        "session_id": session_id,
        "total_intrusions": len(intrusions),
        "zones_breached": len(zone_groups),
        "zones": zones_summary,
    }


def compute_vehicle_frequency_map() -> List[Dict]:
    rows = db_execute(
        "SELECT plate_number, seen_count, avg_speed_kph, vehicle_type, vehicle_color, last_seen FROM vehicles ORDER BY seen_count DESC LIMIT 100",
        fetch="all"
    ) or []
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime.datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


def detect_unusual_vehicle(plate: str) -> Dict:
    info = known_plate_registry.get(plate, {})
    issues = []
    if info.get("flagged"):
        issues.append("flagged_plate")
    if info.get("stolen"):
        issues.append("stolen_vehicle")
    recent_count = detect_repetitive_plate(plate, window_minutes=5)
    if recent_count >= 3:
        issues.append(f"high_frequency_{recent_count}_reads_5min")
    watchlist = check_plate_on_watchlists(plate)
    if watchlist:
        issues.append(f"on_{len(watchlist)}_watchlists")
    return {
        "plate": plate,
        "issues": issues,
        "risk_level": "critical" if len(issues) >= 2 else "high" if issues else "none",
        "vehicle_id": str(info.get("id", "")) if info.get("id") else None,
    }


def build_comparative_session_stats(session_ids: List[str]) -> List[Dict]:
    result = []
    for sid in session_ids:
        alerts = db_execute("SELECT COUNT(*) FROM alerts WHERE session_id = %s", (sid,), fetch="scalar") or 0
        detections = db_execute("SELECT COUNT(*) FROM detections WHERE session_id = %s", (sid,), fetch="scalar") or 0
        persons = db_execute("SELECT COUNT(DISTINCT person_id) FROM detections WHERE session_id = %s AND person_id IS NOT NULL", (sid,), fetch="scalar") or 0
        vehicles = db_execute("SELECT COUNT(DISTINCT vehicle_id) FROM detections WHERE session_id = %s AND vehicle_id IS NOT NULL", (sid,), fetch="scalar") or 0
        critical = db_execute("SELECT COUNT(*) FROM alerts WHERE session_id = %s AND severity = 'critical'", (sid,), fetch="scalar") or 0
        result.append({
            "session_id": sid,
            "total_alerts": int(alerts),
            "total_detections": int(detections),
            "unique_persons": int(persons),
            "unique_vehicles": int(vehicles),
            "critical_alerts": int(critical),
        })
    return result


def compute_threat_escalation_rate(session_id: str) -> Dict:
    rows = db_execute(
        "SELECT created_at, severity FROM alerts WHERE session_id = %s ORDER BY created_at ASC",
        (session_id,), fetch="all"
    ) or []
    if len(rows) < 2:
        return {"session_id": session_id, "escalation_rate": 0, "trend": "stable"}
    severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    scores = [severity_scores.get(r["severity"], 0) for r in rows]
    first_half = np.mean(scores[:len(scores)//2]) if len(scores) > 1 else 0
    second_half = np.mean(scores[len(scores)//2:]) if len(scores) > 1 else 0
    rate = round(float(second_half - first_half), 3)
    trend = "escalating" if rate > 0.3 else "de-escalating" if rate < -0.3 else "stable"
    return {
        "session_id": session_id,
        "escalation_rate": rate,
        "trend": trend,
        "avg_severity_first_half": round(float(first_half), 3),
        "avg_severity_second_half": round(float(second_half), 3),
    }


def auto_generate_watch_alert(person_id: str, session_id: str):
    watchlists = check_person_on_watchlists(person_id)
    if not watchlists:
        return
    for wl in watchlists:
        alert = AlertResult(
            alert_type="watchlist_person_detected",
            severity="critical" if wl.get("priority", 1) >= 3 else "high",
            description=f"Watchlist hit: {wl.get('list_name')} — {wl.get('reason', 'no reason')}",
            person_id=person_id,
            metadata={"watchlist": wl}
        )
        db_save_alert(alert, session_id, None)


def compute_detection_gap_analysis(session_id: str) -> Dict:
    rows = db_execute(
        "SELECT detected_at FROM detections WHERE session_id = %s ORDER BY detected_at ASC",
        (session_id,), fetch="all"
    ) or []
    if len(rows) < 2:
        return {"session_id": session_id, "gaps": [], "max_gap_seconds": 0}
    gaps = []
    for i in range(1, len(rows)):
        a = rows[i-1]["detected_at"]
        b = rows[i]["detected_at"]
        if isinstance(a, datetime.datetime) and isinstance(b, datetime.datetime):
            diff = (b - a).total_seconds()
            if diff > 10:
                gaps.append({"gap_seconds": round(diff, 1), "from": a.isoformat(), "to": b.isoformat()})
    return {
        "session_id": session_id,
        "total_gaps": len(gaps),
        "max_gap_seconds": max((g["gap_seconds"] for g in gaps), default=0),
        "gaps": gaps[:20],
    }


def build_operator_leaderboard() -> List[Dict]:
    rows = db_execute(
        """SELECT operator,
           COUNT(DISTINCT id) as sessions,
           SUM(total_alerts) as total_alerts,
           SUM(total_persons_unique) as total_persons,
           AVG(avg_fps) as avg_fps
           FROM sessions WHERE operator IS NOT NULL
           GROUP BY operator ORDER BY total_alerts DESC LIMIT 20""",
        fetch="all"
    ) or []
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if v is not None and not isinstance(v, str):
                try:
                    d[k] = round(float(v), 2) if "." in str(v) else int(v)
                except Exception:
                    pass
        result.append(d)
    return result


def compute_multi_person_overlap(person_ids: List[str], session_id: str) -> Dict:
    overlap_data = {}
    for pid in person_ids:
        rows = db_execute(
            "SELECT detected_at FROM detections WHERE person_id = %s AND session_id = %s ORDER BY detected_at ASC",
            (pid, session_id), fetch="all"
        ) or []
        overlap_data[pid] = [r["detected_at"] for r in rows if isinstance(r.get("detected_at"), datetime.datetime)]
    co_presence = {}
    pids = list(overlap_data.keys())
    for i in range(len(pids)):
        for j in range(i+1, len(pids)):
            pa, pb = pids[i], pids[j]
            times_a = set(t.strftime("%Y-%m-%dT%H:%M") for t in overlap_data[pa])
            times_b = set(t.strftime("%Y-%m-%dT%H:%M") for t in overlap_data[pb])
            overlap_minutes = len(times_a & times_b)
            co_presence[f"{pa[:8]}__{pb[:8]}"] = {
                "overlap_minutes": overlap_minutes,
                "person_a": pa,
                "person_b": pb,
            }
    return {"session_id": session_id, "co_presence": co_presence}


def compute_face_reappearance_rate(days: int = 7) -> Dict:
    since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    total = db_execute("SELECT COUNT(*) FROM persons WHERE last_seen >= %s", (since,), fetch="scalar") or 0
    returning = db_execute("SELECT COUNT(*) FROM persons WHERE last_seen >= %s AND seen_count > 1", (since,), fetch="scalar") or 0
    rate = round(int(returning) / max(int(total), 1) * 100, 2)
    return {
        "period_days": days,
        "total_active_persons": int(total),
        "returning_persons": int(returning),
        "reappearance_rate_pct": rate,
    }


def build_shift_handoff_report(session_id: str, outgoing_operator: str, incoming_operator: str) -> Dict:
    s_row = db_execute("SELECT * FROM sessions WHERE id = %s", (session_id,), fetch="one")
    session_data = dict(s_row) if s_row else {}
    for k, v in session_data.items():
        if isinstance(v, datetime.datetime):
            session_data[k] = v.isoformat()
    open_alerts = db_execute(
        "SELECT id, alert_type, severity, description, created_at FROM alerts WHERE session_id = %s AND resolved = FALSE ORDER BY severity DESC LIMIT 20",
        (session_id,), fetch="all"
    ) or []
    flagged_persons = db_execute(
        "SELECT p.id, p.name, p.flag_reason, p.threat_level FROM persons p JOIN detections d ON d.person_id = p.id WHERE d.session_id = %s AND p.flagged = TRUE GROUP BY p.id LIMIT 10",
        (session_id,), fetch="all"
    ) or []
    flagged_vehicles = db_execute(
        "SELECT v.id, v.plate_number, v.flag_reason FROM vehicles v JOIN detections d ON d.vehicle_id = v.id WHERE d.session_id = %s AND v.flagged = TRUE GROUP BY v.id LIMIT 10",
        (session_id,), fetch="all"
    ) or []
    def safe(rows):
        result = []
        for r in rows:
            d = dict(r)
            for k, v in d.items():
                if isinstance(v, datetime.datetime):
                    d[k] = v.isoformat()
            result.append(d)
        return result
    return {
        "report_type": "shift_handoff",
        "session_id": session_id,
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "outgoing_operator": outgoing_operator,
        "incoming_operator": incoming_operator,
        "session_summary": session_data,
        "open_alerts": safe(open_alerts),
        "open_alert_count": len(open_alerts),
        "flagged_persons_on_site": safe(flagged_persons),
        "flagged_vehicles_on_site": safe(flagged_vehicles),
        "zone_status": build_zone_report(session_id),
        "threat_trend": compute_threat_escalation_rate(session_id),
    }


@app.get("/persons/{person_id}/dwell-time/{session_id}", tags=["persons"])
async def person_dwell_time(person_id: str, session_id: str, user=Depends(require_auth)):
    return compute_person_dwell_time(person_id, session_id)


@app.get("/persons/{person_id}/pattern-of-life", tags=["persons"])
async def person_pattern_of_life(person_id: str, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, detect_pattern_of_life, person_id)


@app.get("/sessions/ranked/threat", tags=["analytics"])
async def sessions_ranked_by_threat(limit: int = 10, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, rank_sessions_by_threat, limit)


@app.post("/sessions/compare", tags=["sessions"])
async def compare_sessions(session_ids: List[str], user=Depends(require_auth)):
    if len(session_ids) > 10:
        raise HTTPException(status_code=400, detail="Max 10 sessions for comparison")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, build_comparative_session_stats, session_ids)


@app.get("/sessions/{session_id}/zone-report", tags=["sessions"])
async def session_zone_report(session_id: str, user=Depends(require_auth)):
    return build_zone_report(session_id)


@app.get("/sessions/{session_id}/threat-trend", tags=["sessions"])
async def session_threat_trend(session_id: str, user=Depends(require_auth)):
    return compute_threat_escalation_rate(session_id)


@app.get("/sessions/{session_id}/detection-gaps", tags=["sessions"])
async def session_detection_gaps(session_id: str, user=Depends(require_auth)):
    return compute_detection_gap_analysis(session_id)


@app.get("/sessions/{session_id}/camera-coverage", tags=["sessions"])
async def session_camera_coverage(session_id: str, user=Depends(require_auth)):
    return compute_camera_coverage_score(session_id)


@app.post("/sessions/{session_id}/handoff-report", tags=["sessions"])
async def session_handoff_report(
    session_id: str,
    outgoing: str = Query(...),
    incoming: str = Query(...),
    user=Depends(require_auth)
):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, build_shift_handoff_report, session_id, outgoing, incoming)


@app.post("/analyze/multi-person-overlap", tags=["analysis"])
async def multi_person_overlap(session_id: str, person_ids: List[str], user=Depends(require_auth)):
    if len(person_ids) > 20:
        raise HTTPException(status_code=400, detail="Max 20 persons")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, compute_multi_person_overlap, person_ids, session_id)


@app.get("/analyze/vehicle/{plate}", tags=["analysis"])
async def analyze_vehicle_plate(plate: str, user=Depends(require_auth)):
    return detect_unusual_vehicle(plate.upper())


@app.get("/stats/vehicles/frequency-map", tags=["analytics"])
async def vehicle_frequency_map(user=Depends(require_auth)):
    return {"vehicles": compute_vehicle_frequency_map()}


@app.get("/stats/faces/reappearance-rate", tags=["analytics"])
async def face_reappearance_rate(days: int = 7, user=Depends(require_auth)):
    return compute_face_reappearance_rate(days)


@app.get("/stats/sessions/threat-ranking", tags=["analytics"])
async def session_threat_ranking(limit: int = 10, user=Depends(require_auth)):
    return await sessions_ranked_by_threat(limit, user)


@app.get("/admin/operators/leaderboard", tags=["admin"])
async def operator_leaderboard(user=Depends(require_admin)):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, build_operator_leaderboard)


@app.get("/export/handoff/{session_id}", tags=["export"])
async def export_handoff_report(
    session_id: str,
    outgoing: str = Query(...),
    incoming: str = Query(...),
    user=Depends(require_auth)
):
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(executor, build_shift_handoff_report, session_id, outgoing, incoming)
    out = json.dumps(report, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(out.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=argus_handoff_{session_id[:8]}.json"}
    )


@app.get("/export/pattern-of-life/{person_id}", tags=["export"])
async def export_pattern_of_life(person_id: str, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, detect_pattern_of_life, person_id)
    out = json.dumps(data, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(out.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=argus_pol_{person_id[:8]}.json"}
    )


def compute_real_time_threat_index(session_id: str) -> Dict:
    s = active_sessions.get(session_id, {})
    alert_count = s.get("alert_count", 0)
    unique_persons = len(s.get("unique_persons", set()))
    fps = s.get("fps", 0)
    recent_critical = db_execute(
        "SELECT COUNT(*) FROM alerts WHERE session_id = %s AND severity = 'critical' AND created_at > NOW() - INTERVAL '5 minutes'",
        (session_id,), fetch="scalar"
    ) or 0
    recent_flagged = db_execute(
        "SELECT COUNT(*) FROM detections WHERE session_id = %s AND is_flagged = TRUE AND detected_at > NOW() - INTERVAL '5 minutes'",
        (session_id,), fetch="scalar"
    ) or 0
    index = 0.0
    index += min(int(recent_critical) * 25.0, 50.0)
    index += min(int(recent_flagged) * 15.0, 30.0)
    index += min(alert_count * 0.5, 10.0)
    index += min(unique_persons * 0.2, 10.0)
    index = min(round(index, 2), 100.0)
    level = "critical" if index >= 70 else "high" if index >= 40 else "medium" if index >= 20 else "low"
    return {
        "session_id": session_id,
        "threat_index": index,
        "threat_level": level,
        "recent_critical_alerts_5m": int(recent_critical),
        "recent_flagged_detections_5m": int(recent_flagged),
        "total_alerts": alert_count,
        "active_persons": unique_persons,
        "fps": fps,
        "computed_at": datetime.datetime.utcnow().isoformat(),
    }


def compute_night_activity_report(session_id: str) -> Dict:
    rows = db_execute(
        "SELECT COUNT(*) as count, AVG(threat_score) as avg_threat FROM detections WHERE session_id = %s AND night_mode = TRUE",
        (session_id,), fetch="one"
    )
    total_night = int(rows["count"]) if rows else 0
    avg_threat = round(float(rows["avg_threat"]), 3) if rows and rows["avg_threat"] else 0.0
    alerts_night = db_execute(
        "SELECT COUNT(*) FROM alerts WHERE session_id = %s AND metadata->>'night_mode' = 'true'",
        (session_id,), fetch="scalar"
    ) or 0
    return {
        "session_id": session_id,
        "night_detections": total_night,
        "avg_night_threat_score": avg_threat,
        "night_alerts": int(alerts_night),
        "night_active_currently": night_mode_active.get(session_id, False),
    }


def compute_loitering_map(session_id: str) -> List[Dict]:
    entries = []
    for key, ts in list(loiter_tracker.items()):
        if key.startswith(session_id):
            duration = round(time.time() - ts, 1)
            parts = key.split("_")
            track_id = int(parts[-1]) if parts[-1].isdigit() else -1
            pos = track_position_histories.get(track_id, [])
            last_pos = pos[-1] if pos else None
            entries.append({
                "track_id": track_id,
                "loiter_duration_seconds": duration,
                "position": last_pos,
                "alert_threshold": LOITER_TIME_SECONDS,
                "past_threshold": duration > LOITER_TIME_SECONDS,
            })
    entries.sort(key=lambda x: x["loiter_duration_seconds"], reverse=True)
    return entries


def compute_audio_spectral_summary(session_id: str) -> Dict:
    rows = db_execute(
        "SELECT event_type, AVG(energy_level) as avg_energy, AVG(frequency_peak) as avg_freq, COUNT(*) as count FROM audio_events WHERE session_id = %s GROUP BY event_type ORDER BY count DESC",
        (session_id,), fetch="all"
    ) or []
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if v is not None and not isinstance(v, str):
                try:
                    d[k] = round(float(v), 4)
                except Exception:
                    pass
        result.append(d)
    return {"session_id": session_id, "audio_summary": result}


def build_complete_surveillance_report(session_id: str) -> Dict:
    coverage = compute_session_coverage(session_id)
    zone_rep = build_zone_report(session_id)
    threat_trend = compute_threat_escalation_rate(session_id)
    peak = compute_session_peak_activity(session_id)
    face_acc = compute_face_recognition_accuracy(session_id)
    conf_stats = compute_detection_confidence_stats(session_id)
    gap_analysis = compute_detection_gap_analysis(session_id)
    cam_coverage = compute_camera_coverage_score(session_id)
    demo = build_demographic_report(session_id)
    incident_tl = build_incident_timeline(session_id)
    audio_summary = compute_audio_spectral_summary(session_id)
    unusual_hours = detect_unusual_time_activity(session_id)
    night_report = compute_night_activity_report(session_id)
    loiter_map = compute_loitering_map(session_id)
    plate_conf = compute_plate_confidence_distribution(session_id)
    return {
        "report_type": "complete_surveillance",
        "session_id": session_id,
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "system": "ARGUS v2.0.0",
        "coverage": coverage,
        "zone_report": zone_rep,
        "threat_trend": threat_trend,
        "peak_activity": peak,
        "face_recognition": face_acc,
        "confidence_stats": conf_stats,
        "detection_gaps": gap_analysis,
        "camera_coverage": cam_coverage,
        "demographics": demo,
        "incident_timeline": incident_tl,
        "audio_analysis": audio_summary,
        "unusual_hour_activity": unusual_hours,
        "night_mode_report": night_report,
        "active_loitering": loiter_map,
        "plate_ocr_quality": plate_conf,
    }


@app.get("/sessions/{session_id}/threat-index", tags=["sessions"])
async def session_threat_index(session_id: str, user=Depends(require_auth)):
    return compute_real_time_threat_index(session_id)


@app.get("/sessions/{session_id}/night-report", tags=["sessions"])
async def session_night_report(session_id: str, user=Depends(require_auth)):
    return compute_night_activity_report(session_id)


@app.get("/sessions/{session_id}/loitering-map", tags=["sessions"])
async def session_loitering_map(session_id: str, user=Depends(require_auth)):
    return {"session_id": session_id, "loitering": compute_loitering_map(session_id)}


@app.get("/sessions/{session_id}/audio-summary", tags=["sessions"])
async def session_audio_summary(session_id: str, user=Depends(require_auth)):
    return compute_audio_spectral_summary(session_id)


@app.get("/sessions/{session_id}/complete-report", tags=["sessions"])
async def session_complete_report(session_id: str, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, build_complete_surveillance_report, session_id)


@app.get("/export/complete-report/{session_id}", tags=["export"])
async def export_complete_report(session_id: str, user=Depends(require_auth)):
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(executor, build_complete_surveillance_report, session_id)
    out = json.dumps(report, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(out.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=argus_complete_{session_id[:8]}.json"}
    )


@app.get("/live/threat-indices", tags=["live"])
async def live_threat_indices(user=Depends(require_auth)):
    indices = []
    for sid in list(active_sessions.keys()):
        idx = compute_real_time_threat_index(sid)
        indices.append(idx)
    indices.sort(key=lambda x: x["threat_index"], reverse=True)
    return {"sessions": indices, "count": len(indices)}


@app.get("/admin/full-system-report", tags=["admin"])
async def full_system_report(user=Depends(require_admin)):
    db_stats = {}
    tables = ["persons","vehicles","detections","alerts","plate_reads","audio_events","behavior_events","sessions","watchlist","gait_profiles"]
    for table in tables:
        try:
            c = db_execute(f"SELECT COUNT(*) FROM {table}", fetch="scalar")
            db_stats[table] = int(c or 0)
        except Exception:
            db_stats[table] = -1
    return {
        "system": "ARGUS v2.0.0",
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "database_tables": db_stats,
        "memory": {
            "known_faces": len(known_face_embeddings),
            "flagged_cache": len(flagged_faces_cache),
            "known_plates": len(known_plate_registry),
            "active_sessions": len(active_sessions),
            "active_ws": len(active_ws_connections),
            "active_audio_ws": len(audio_ws_connections),
            "track_ids": len(track_position_histories),
            "loiter_keys": len(loiter_tracker),
            "audio_buffer": len(audio_event_buffer),
            "heatmap_sessions": len(attention_heatmap),
            "fight_buffers": len(fight_motion_buffer),
        },
        "active_sessions": get_active_session_summary(),
        "threat_indices": [compute_real_time_threat_index(sid) for sid in list(active_sessions.keys())],
        "global_stats": {
            "total_persons": db_execute("SELECT COUNT(*) FROM persons", fetch="scalar"),
            "flagged_persons": db_execute("SELECT COUNT(*) FROM persons WHERE flagged=TRUE", fetch="scalar"),
            "total_vehicles": db_execute("SELECT COUNT(*) FROM vehicles", fetch="scalar"),
            "open_alerts": db_execute("SELECT COUNT(*) FROM alerts WHERE resolved=FALSE", fetch="scalar"),
            "critical_open": db_execute("SELECT COUNT(*) FROM alerts WHERE resolved=FALSE AND severity='critical'", fetch="scalar"),
        }
    }

