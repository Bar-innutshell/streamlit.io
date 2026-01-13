"""
� Smart Environment Monitor
Real-time environment monitoring dashboard with ML Prediction
by Kita pergi hari ini
"""

import streamlit as st
import paho.mqtt.client as mqtt
import json
import time
import queue
import threading
import numpy as np
import joblib
from datetime import datetime, timezone, timedelta
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ==================== TERMINAL LOGGER ====================
class TerminalLogger:
    """Enhanced terminal logging with colors and formatting for Testing & Validation"""
    
    # ANSI Color Codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Box characters
    H_LINE = "═"
    V_LINE = "║"
    TL_CORNER = "╔"
    TR_CORNER = "╗"
    BL_CORNER = "╚"
    BR_CORNER = "╝"
    T_LEFT = "╠"
    T_RIGHT = "╣"
    
    BOX_WIDTH = 65
    
    @staticmethod
    def _get_timestamp():
        wib = timezone(timedelta(hours=7))
        return datetime.now(wib).strftime("%H:%M:%S")
    
    @staticmethod
    def _strip_ansi(text):
        """Remove ANSI codes for length calculation"""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', text)
    
    @staticmethod
    def _make_line(content, width=65):
        """Create a formatted line with borders"""
        content_len = len(TerminalLogger._strip_ansi(content))
        padding = width - content_len - 2
        return f"{TerminalLogger.V_LINE} {content}{' ' * max(0, padding)}{TerminalLogger.V_LINE}"
    
    @staticmethod
    def _top_border(width=65):
        return f"{TerminalLogger.TL_CORNER}{TerminalLogger.H_LINE * width}{TerminalLogger.TR_CORNER}"
    
    @staticmethod
    def _bottom_border(width=65):
        return f"{TerminalLogger.BL_CORNER}{TerminalLogger.H_LINE * width}{TerminalLogger.BR_CORNER}"
    
    @staticmethod
    def _mid_border(width=65):
        return f"{TerminalLogger.T_LEFT}{TerminalLogger.H_LINE * width}{TerminalLogger.T_RIGHT}"
    
    @staticmethod
    def _get_temp_category(temp):
        if temp < 20:
            return "Dingin", TerminalLogger.CYAN
        elif temp < 28:
            return "Normal", TerminalLogger.GREEN
        elif temp < 35:
            return "Panas", TerminalLogger.YELLOW
        else:
            return "Sangat Panas", TerminalLogger.RED
    
    @staticmethod
    def _get_humid_category(humid):
        if humid < 40:
            return "Kering", TerminalLogger.YELLOW
        elif humid < 70:
            return "Normal", TerminalLogger.GREEN
        else:
            return "Lembab", TerminalLogger.CYAN
    
    @staticmethod
    def _get_aqi_category(aqi):
        if aqi <= 50:
            return "Baik", TerminalLogger.GREEN
        elif aqi <= 100:
            return "Sedang", TerminalLogger.YELLOW
        elif aqi <= 150:
            return "Tidak Sehat", TerminalLogger.YELLOW
        elif aqi <= 200:
            return "Buruk", TerminalLogger.RED
        else:
            return "Berbahaya", TerminalLogger.RED
    
    @staticmethod
    def _get_rssi_category(rssi):
        if rssi > -50:
            return "Excellent", TerminalLogger.GREEN
        elif rssi > -60:
            return "Good", TerminalLogger.GREEN
        elif rssi > -70:
            return "Fair", TerminalLogger.YELLOW
        else:
            return "Weak", TerminalLogger.RED
    
    @staticmethod
    def sensor_data(suhu, lembab, aqi, rssi=None, adc_raw=None):
        """Log formatted sensor data"""
        ts = TerminalLogger._get_timestamp()
        w = TerminalLogger.BOX_WIDTH
        
        temp_cat, temp_color = TerminalLogger._get_temp_category(suhu)
        humid_cat, humid_color = TerminalLogger._get_humid_category(lembab)
        aqi_cat, aqi_color = TerminalLogger._get_aqi_category(aqi)
        
        print()
        print(f"{TerminalLogger.CYAN}{TerminalLogger._top_border(w)}{TerminalLogger.RESET}")
        header = f"{TerminalLogger.BOLD}📊 SENSOR DATA UPDATE [{ts}]{TerminalLogger.RESET}"
        print(f"{TerminalLogger.CYAN}{TerminalLogger._make_line(header, w)}{TerminalLogger.RESET}")
        print(f"{TerminalLogger.CYAN}{TerminalLogger._mid_border(w)}{TerminalLogger.RESET}")
        
        # Temperature
        temp_line = f"🌡️  Suhu       : {temp_color}{suhu:>6.1f}°C{TerminalLogger.RESET}  ({temp_cat})"
        print(f"{TerminalLogger.CYAN}{TerminalLogger._make_line(temp_line, w)}{TerminalLogger.RESET}")
        
        # Humidity
        humid_line = f"💧 Kelembaban : {humid_color}{lembab:>6.1f}%{TerminalLogger.RESET}   ({humid_cat})"
        print(f"{TerminalLogger.CYAN}{TerminalLogger._make_line(humid_line, w)}{TerminalLogger.RESET}")
        
        # AQI
        aqi_line = f"💨 AQI        : {aqi_color}{aqi:>6}{TerminalLogger.RESET}     ({aqi_cat})"
        print(f"{TerminalLogger.CYAN}{TerminalLogger._make_line(aqi_line, w)}{TerminalLogger.RESET}")
        
        # ADC Raw (if provided)
        if adc_raw is not None:
            adc_line = f"📈 ADC Raw    : {adc_raw:>6}"
            print(f"{TerminalLogger.CYAN}{TerminalLogger._make_line(adc_line, w)}{TerminalLogger.RESET}")
        
        # WiFi RSSI (if provided)
        if rssi is not None:
            rssi_cat, rssi_color = TerminalLogger._get_rssi_category(rssi)
            rssi_line = f"📶 WiFi RSSI  : {rssi_color}{rssi:>6} dBm{TerminalLogger.RESET} ({rssi_cat})"
            print(f"{TerminalLogger.CYAN}{TerminalLogger._make_line(rssi_line, w)}{TerminalLogger.RESET}")
        
        print(f"{TerminalLogger.CYAN}{TerminalLogger._bottom_border(w)}{TerminalLogger.RESET}")
    
    @staticmethod
    def ml_prediction(status, confidence, suhu=None, lembab=None, aqi=None, sent_ok=True):
        """Log ML prediction result"""
        ts = TerminalLogger._get_timestamp()
        w = TerminalLogger.BOX_WIDTH
        
        # Color based on prediction
        if status.lower() in ["aman", "baik", "good"]:
            status_color = TerminalLogger.GREEN
            icon = "✅"
        elif status.lower() in ["waspada", "hati-hati", "sedang", "moderate"]:
            status_color = TerminalLogger.YELLOW
            icon = "⚠️"
        else:
            status_color = TerminalLogger.RED
            icon = "🚨"
        
        print()
        print(f"{TerminalLogger.MAGENTA}{TerminalLogger._top_border(w)}{TerminalLogger.RESET}")
        header = f"{TerminalLogger.BOLD}🤖 ML PREDICTION [{ts}]{TerminalLogger.RESET}"
        print(f"{TerminalLogger.MAGENTA}{TerminalLogger._make_line(header, w)}{TerminalLogger.RESET}")
        print(f"{TerminalLogger.MAGENTA}{TerminalLogger._mid_border(w)}{TerminalLogger.RESET}")
        
        # Input features (if provided)
        if suhu is not None and lembab is not None and aqi is not None:
            input_line = f"Input  : [{suhu:.1f}°C, {lembab:.1f}%, AQI {aqi}]"
            print(f"{TerminalLogger.MAGENTA}{TerminalLogger._make_line(input_line, w)}{TerminalLogger.RESET}")
        
        # Result
        conf_str = f"{confidence:.1f}%" if confidence else "N/A"
        result_line = f"Result : {status_color}{TerminalLogger.BOLD}{icon} {status}{TerminalLogger.RESET} (Confidence: {conf_str})"
        print(f"{TerminalLogger.MAGENTA}{TerminalLogger._make_line(result_line, w)}{TerminalLogger.RESET}")
        
        # Sent status
        sent_status = f"{TerminalLogger.GREEN}Sent ✓{TerminalLogger.RESET}" if sent_ok else f"{TerminalLogger.RED}Failed ✗{TerminalLogger.RESET}"
        sent_line = f"ESP32  : {sent_status}"
        print(f"{TerminalLogger.MAGENTA}{TerminalLogger._make_line(sent_line, w)}{TerminalLogger.RESET}")
        
        print(f"{TerminalLogger.MAGENTA}{TerminalLogger._bottom_border(w)}{TerminalLogger.RESET}")
    
    @staticmethod
    def mqtt_status(event, broker=None, port=None, topic=None, rc=None):
        """Log MQTT connection status"""
        ts = TerminalLogger._get_timestamp()
        
        if event == "connected":
            print(f"\n{TerminalLogger.GREEN}{TerminalLogger.BOLD}[{ts}] ✅ MQTT CONNECTED{TerminalLogger.RESET}")
            print(f"  └─ Broker: {broker}:{port}")
            if topic:
                print(f"  └─ Topics: {topic}")
        elif event == "disconnected":
            print(f"\n{TerminalLogger.YELLOW}{TerminalLogger.BOLD}[{ts}] ⚠️ MQTT DISCONNECTED{TerminalLogger.RESET}")
            if rc is not None:
                print(f"  └─ Reason code: {rc}")
        elif event == "connecting":
            print(f"\n{TerminalLogger.BLUE}[{ts}] 🔌 MQTT Connecting to {broker}:{port}...{TerminalLogger.RESET}")
        elif event == "error":
            print(f"\n{TerminalLogger.RED}{TerminalLogger.BOLD}[{ts}] ❌ MQTT ERROR{TerminalLogger.RESET}")
            if rc:
                print(f"  └─ Error: {rc}")
        elif event == "subscribed":
            print(f"{TerminalLogger.DIM}[{ts}] 📥 Subscribed: {topic}{TerminalLogger.RESET}")
    
    @staticmethod
    def actuator_response(data):
        """Log ESP32 actuator response"""
        ts = TerminalLogger._get_timestamp()
        w = TerminalLogger.BOX_WIDTH
        
        print()
        print(f"{TerminalLogger.YELLOW}{TerminalLogger._top_border(w)}{TerminalLogger.RESET}")
        header = f"{TerminalLogger.BOLD}⚙️ ESP32 RESPONSE [{ts}]{TerminalLogger.RESET}"
        print(f"{TerminalLogger.YELLOW}{TerminalLogger._make_line(header, w)}{TerminalLogger.RESET}")
        print(f"{TerminalLogger.YELLOW}{TerminalLogger._mid_border(w)}{TerminalLogger.RESET}")
        
        if isinstance(data, dict):
            for key, value in data.items():
                line = f"{key}: {value}"
                print(f"{TerminalLogger.YELLOW}{TerminalLogger._make_line(line, w)}{TerminalLogger.RESET}")
        else:
            print(f"{TerminalLogger.YELLOW}{TerminalLogger._make_line(str(data), w)}{TerminalLogger.RESET}")
        
        print(f"{TerminalLogger.YELLOW}{TerminalLogger._bottom_border(w)}{TerminalLogger.RESET}")
    
    @staticmethod
    def command_sent(cmd, **kwargs):
        """Log command sent to ESP32"""
        ts = TerminalLogger._get_timestamp()
        details = ", ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        print(f"{TerminalLogger.BLUE}[{ts}] 📤 Command: {cmd} {details}{TerminalLogger.RESET}")
    
    @staticmethod
    def info(message):
        """Log info message"""
        ts = TerminalLogger._get_timestamp()
        print(f"{TerminalLogger.CYAN}[{ts}] ℹ️ {message}{TerminalLogger.RESET}")
    
    @staticmethod
    def success(message):
        """Log success message"""
        ts = TerminalLogger._get_timestamp()
        print(f"{TerminalLogger.GREEN}[{ts}] ✅ {message}{TerminalLogger.RESET}")
    
    @staticmethod
    def warning(message):
        """Log warning message"""
        ts = TerminalLogger._get_timestamp()
        print(f"{TerminalLogger.YELLOW}[{ts}] ⚠️ {message}{TerminalLogger.RESET}")
    
    @staticmethod
    def error(message):
        """Log error message"""
        ts = TerminalLogger._get_timestamp()
        print(f"{TerminalLogger.RED}[{ts}] ❌ {message}{TerminalLogger.RESET}")

# Create global logger instance
logger = TerminalLogger()

# Page config

st.set_page_config(
    page_title="Smart Environment Monitor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== ML CONFIGURATION ====================
MODEL_PATH = "model_svm_rbf.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
TOPIC_PREDICTION = "projek/asma/prediction"
MAX_POINTS = 100

# ==================== LOAD ML MODEL ====================
import os
import tempfile

@st.cache_resource
def load_ml_model(_model_path=None):
    """Load ML model from path. Accepts custom path or uses default."""
    model_path = _model_path if _model_path else MODEL_PATH
    
    if not os.path.exists(model_path):
        return None, None, None, f"⚠️ Model file not found: {os.path.basename(model_path)}"
    try:
        model = joblib.load(model_path)
        
        # Load scaler for SVM
        scaler = None
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        
        # Load label encoder
        label_encoder = None
        if os.path.exists(LABEL_ENCODER_PATH):
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        return model, scaler, label_encoder, None
    except Exception as e:
        error_msg = str(e)
        if "incompatible dtype" in error_msg or "node array" in error_msg:
            return None, None, None, "⚠️ Model incompatible! Install scikit-learn==1.1.3: pip install scikit-learn==1.1.3"
        return None, None, None, f"❌ Failed to load model: {error_msg}"

# Initialize model path in session state
if 'custom_model_path' not in st.session_state:
    st.session_state.custom_model_path = None

# Load model (custom or default)
if st.session_state.custom_model_path:
    ml_model, ml_scaler, ml_label_encoder, ml_model_error = load_ml_model(st.session_state.custom_model_path)
else:
    ml_model, ml_scaler, ml_label_encoder, ml_model_error = load_ml_model()

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Theme colors - Professional teal/orange palette
THEMES = {
    'dark': {
        'bg_primary': '#0d1117',
        'bg_secondary': '#161b22',
        'bg_card': 'linear-gradient(145deg, #1c2128, #21262d)',
        'bg_sidebar': 'linear-gradient(180deg, #1c2128 0%, #0d1117 100%)',
        'text_primary': '#e6edf3',
        'text_secondary': '#8b949e',
        'accent': '#2f9e8f',
        'accent_secondary': '#f0883e',
        'accent_gradient': 'linear-gradient(90deg, #2f9e8f, #58d1c9)',
        'success': '#3fb950',
        'warning': '#d29922',
        'danger': '#f85149',
        'border': 'rgba(48, 54, 61, 0.8)',
        'chart_bg': 'rgba(0,0,0,0.1)',
        'grid': 'rgba(255,255,255,0.08)'
    },
    'light': {
        'bg_primary': '#ffffff',
        'bg_secondary': '#f6f8fa',
        'bg_card': 'linear-gradient(145deg, #ffffff, #f6f8fa)',
        'bg_sidebar': 'linear-gradient(180deg, #f6f8fa 0%, #e8ebef 100%)',
        'text_primary': '#1f2328',
        'text_secondary': '#656d76',
        'accent': '#0969da',
        'accent_secondary': '#bf8700',
        'accent_gradient': 'linear-gradient(90deg, #0969da, #54aeff)',
        'success': '#1a7f37',
        'warning': '#9a6700',
        'danger': '#cf222e',
        'border': 'rgba(208, 215, 222, 1)',
        'chart_bg': 'rgba(255,255,255,0.95)',
        'grid': 'rgba(0,0,0,0.1)'
    }
}

# SVG Icons
SVG_ICONS = {
    'temperature': '''<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/></svg>''',
    'humidity': '''<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></svg>''',
    'gas': '''<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>''',
    'wifi_strong': '''<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>''',
    'wifi_weak': '''<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>''',
    'alert': '''<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>''',
    'fire': '''<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>''',
    'chart': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>''',
    'table': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18"/><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/></svg>''',
    'gauge': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>''',
    'settings': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>''',
    'moon': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>''',
    'sun': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>''',
    'plug': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22v-5"/><path d="M9 8V2"/><path d="M15 8V2"/><path d="M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8Z"/></svg>''',
    'info': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>''',
    'activity': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>''',
    'server': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="8" x="2" y="2" rx="2" ry="2"/><rect width="20" height="8" x="2" y="14" rx="2" ry="2"/><line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/></svg>'''
}

def get_theme():
    return THEMES[st.session_state.theme]

theme = get_theme()

# Dynamic CSS based on theme
st.markdown(f"""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for responsive sizing */
    :root {{
        --font-size-xs: 0.75rem;
        --font-size-sm: 0.875rem;
        --font-size-base: 1rem;
        --font-size-lg: 1.125rem;
        --font-size-xl: 1.25rem;
        --font-size-2xl: 1.5rem;
        --font-size-3xl: 2rem;
        --font-size-4xl: 2.5rem;
        --font-size-5xl: 3rem;
        --spacing-xs: 0.25rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
        --spacing-xl: 2rem;
        --card-padding: 18px;
        --card-radius: 16px;
        --metric-value-size: 2.2rem;
        --metric-icon-size: 2.5rem;
        --title-size: 2.5rem;
        --section-title-size: 1.5rem;
        --gauge-height: 280px;
        --chart-height: 650px;
    }}
    
    /* Global font */
    * {{
        font-family: 'Poppins', sans-serif;
    }}
    
    /* Main background */
    .stApp {{
        background: {theme['bg_primary']};
        color: {theme['text_primary']};
    }}
    
    /* Equal height columns for metric cards */
    [data-testid="stHorizontalBlock"] {{
        align-items: stretch !important;
        display: flex !important;
        gap: 16px;
    }}
    
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 0 !important;
        min-width: 0 !important;
    }}
    
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {{
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }}
    
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div > div {{
        flex: 1 !important;
        height: 100% !important;
    }}
    
    /* Sidebar collapse button always visible */
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {{
        opacity: 1 !important;
        visibility: visible !important;
    }}
    
    button[kind="headerNoPadding"] {{
        opacity: 1 !important;
        visibility: visible !important;
    }}
    
    /* ============================================
       TOOLBAR & HEADER STYLING
       ============================================ */
    
    /* App Header & Toolbar */
    [data-testid="stHeader"] {{
        background-color: {theme['bg_primary']} !important;
        border-bottom: 1px solid {theme['border']} !important;
    }}
    
    [data-testid="stToolbar"] {{
        background-color: {theme['bg_primary']} !important;
    }}
    
    /* Toolbar buttons and text */
    [data-testid="stToolbar"] button {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stToolbar"] button span {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stToolbar"] button svg {{
        fill: {theme['text_primary']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    /* Deploy button */
    button[kind="header"] {{
        color: {theme['text_primary']} !important;
        background-color: transparent !important;
    }}
    
    button[kind="header"]:hover {{
        background-color: {theme['bg_secondary']} !important;
    }}
    
    button[kind="header"] span {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Main Menu button */
    [data-testid="stMainMenu"] button {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stMainMenu"] svg {{
        fill: {theme['text_primary']} !important;
    }}
    
    /* Sidebar expand/collapse button */
    button[data-testid="stExpandSidebarButton"] {{
        color: {theme['text_primary']} !important;
    }}
    
    button[data-testid="stExpandSidebarButton"] span {{
        color: {theme['text_primary']} !important;
    }}
    
    button[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    button[data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        min-width: 0 !important;
    }}
    
    /* Custom Header Bar */
    .custom-header {{
        background: {theme['bg_card']};
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 20px;
        border: 1px solid {theme['border']};
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }}
    
    .header-left {{
        display: flex;
        align-items: center;
        gap: 15px;
        flex-wrap: wrap;
    }}
    
    .header-title {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: {theme['text_primary']};
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    .header-status {{
        display: flex;
        align-items: center;
        gap: 15px;
        flex-wrap: wrap;
    }}
    
    .status-badge {{
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }}
    
    .status-online {{
        background: rgba(63, 185, 80, 0.15);
        color: {theme['success']};
        border: 1px solid {theme['success']};
    }}
    
    .status-offline {{
        background: rgba(248, 81, 73, 0.15);
        color: {theme['danger']};
        border: 1px solid {theme['danger']};
    }}
    
    .header-info {{
        color: {theme['text_secondary']};
        font-size: 0.75rem;
    }}
    
    .header-right {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    .theme-toggle-btn {{
        background: {theme['bg_secondary']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        padding: 8px 16px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        color: {theme['text_primary']};
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .theme-toggle-btn:hover {{
        background: {theme['accent']};
        border-color: {theme['accent']};
        color: white;
    }}

    /* Sidebar styling - kept for reference but hidden */
    [data-testid="stSidebar"] {{
        background: {theme['bg_sidebar']};
        border-right: 1px solid {theme['border']};
    }}
    
    [data-testid="stSidebar"] * {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Headers */
    h1, h2, h3 {{
        font-family: 'Poppins', sans-serif !important;
        letter-spacing: 1px;
        font-weight: 600;
    }}
    
    /* Cards - Desktop default - Equal sizing for all cards */
    .metric-card {{
        background: {theme['bg_card']};
        border-radius: var(--card-radius);
        padding: var(--card-padding);
        border: 1px solid {theme['border']};
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin: 8px 0 12px 0;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        height: 185px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(47, 158, 143, 0.2);
        border-color: {theme['accent']};
    }}
    
    .metric-value {{
        font-family: 'Poppins', sans-serif;
        font-size: var(--metric-value-size);
        font-weight: 700;
        color: {theme['text_primary']};
        margin: 6px 0;
        line-height: 1.1;
    }}
    
    .metric-label {{
        font-family: 'Poppins', sans-serif;
        color: {theme['text_secondary']};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
        margin-bottom: 4px;
    }}
    
    .metric-icon {{
        margin-bottom: 8px;
        color: {theme['accent']};
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .metric-icon svg {{
        width: 36px;
        height: 36px;
        stroke: {theme['accent']};
    }}
    
    /* WiFi Signal Card Enhanced */
    .wifi-signal-container {{
        width: 100%;
        padding: 0 10px;
    }}
    
    .wifi-signal-bar {{
        width: 100%;
        height: 8px;
        background: {theme['bg_secondary']};
        border-radius: 10px;
        overflow: hidden;
        margin: 8px 0;
        position: relative;
    }}
    
    .wifi-signal-fill {{
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease, background 0.3s ease;
        position: relative;
    }}
    
    .wifi-signal-fill::after {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}
    
    .wifi-quality-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 6px;
    }}
    
    .wifi-quality-excellent {{
        background: rgba(63, 185, 80, 0.15);
        color: {theme['success']};
        border: 1px solid rgba(63, 185, 80, 0.3);
    }}
    
    .wifi-quality-good {{
        background: rgba(88, 166, 255, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(88, 166, 255, 0.3);
    }}
    
    .wifi-quality-fair {{
        background: rgba(210, 153, 34, 0.15);
        color: {theme['warning']};
        border: 1px solid rgba(210, 153, 34, 0.3);
    }}
    
    .wifi-quality-weak {{
        background: rgba(248, 81, 73, 0.15);
        color: {theme['danger']};
        border: 1px solid rgba(248, 81, 73, 0.3);
    }}
    
    .wifi-dbm-value {{
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: {theme['text_primary']};
        margin: 4px 0;
    }}
    
    .wifi-dbm-unit {{
        font-size: 0.8rem;
        color: {theme['text_secondary']};
        font-weight: 400;
    }}
    
    /* ML Prediction Confidence */
    .ml-confidence {{
        font-size: 0.75rem;
        color: {theme['text_secondary']};
        margin-top: 4px;
        font-weight: 500;
    }}
    
    /* Spacer to equalize card heights */
    .metric-spacer {{
        height: 32px;
    }}
    
    /* Gauge Cards Container - Fixed height for all gauges */
    .gauge-card {{
        background: {theme['bg_card']};
        border-radius: var(--card-radius);
        padding: 15px;
        border: 1px solid {theme['border']};
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        min-height: 280px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin: 8px 0;
        box-sizing: border-box;
    }}
    
    /* Plotly chart container fix */
    .stPlotlyChart {{
        width: 100% !important;
    }}
    
    .stPlotlyChart > div {{
        width: 100% !important;
    }}
    
    .js-plotly-plot {{
        width: 100% !important;
    }}
    
    /* Status badges */
    .status-online {{
        background: {theme['success']};
        color: #fff;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        letter-spacing: 1px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: var(--font-size-sm);
    }}
    
    .status-offline {{
        background: {theme['danger']};
        color: #fff;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        letter-spacing: 1px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: var(--font-size-sm);
    }}
    
    /* Title styling */
    .main-title {{
        font-family: 'Poppins', sans-serif;
        color: {theme['text_primary']};
        font-size: var(--title-size);
        font-weight: 700;
        text-align: center;
        margin-bottom: 40px;
        letter-spacing: 2px;
        line-height: 1.3;
        padding: 0 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
    }}
    
    .main-title svg {{
        stroke: {theme['accent']};
    }}
    
    /* Section titles */
    .section-title {{
        font-family: 'Poppins', sans-serif;
        color: {theme['text_primary']};
        font-size: var(--section-title-size);
        font-weight: 600;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid {theme['border']};
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    .section-title svg {{
        stroke: {theme['accent']};
    }}
    
    /* Alert box */
    .alert-danger {{
        background: {theme['danger']};
        color: white;
        padding: 15px 20px;
        border-radius: 12px;
        text-align: left;
        margin: 15px 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        font-size: var(--font-size-base);
        display: flex;
        align-items: center;
        gap: 12px;
        border-left: 4px solid #a91b24;
    }}
    
    .alert-warning {{
        background: {theme['warning']};
        color: #1a1a1a;
        padding: 15px 20px;
        border-radius: 12px;
        text-align: left;
        margin: 15px 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        font-size: var(--font-size-base);
        display: flex;
        align-items: center;
        gap: 12px;
        border-left: 4px solid #9a6700;
    }}
    
    .alert-success {{
        background: {theme['success']};
        color: white;
        padding: 15px 20px;
        border-radius: 12px;
        text-align: left;
        margin: 15px 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        font-size: var(--font-size-base);
        display: flex;
        align-items: center;
        gap: 12px;
        border-left: 4px solid #116329;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: {theme['accent']};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 25px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
        font-size: var(--font-size-sm);
        width: 100%;
    }}
    
    .stButton > button:hover {{
        opacity: 0.9;
        transform: translateY(-1px);
    }}
    
    /* Input styling */
    .stTextInput > div > div > input {{
        background: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 15px;
        color: {theme['text_primary']} !important;
        font-family: 'Poppins', sans-serif;
        padding: 10px 15px;
        font-size: var(--font-size-sm);
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {theme['accent']} !important;
        box-shadow: 0 0 10px rgba(9, 105, 218, 0.2);
    }}
    
    /* Text input labels */
    .stTextInput label {{
        color: {theme['text_primary']} !important;
        font-weight: 500;
    }}
    
    /* Number input */
    .stNumberInput > div > div > input {{
        background: {theme['bg_secondary']} !important;
        border: 2px solid {theme['border']} !important;
        border-radius: 15px;
        color: {theme['text_primary']} !important;
        font-size: var(--font-size-sm);
    }}
    
    /* Number input + and - buttons styling for light/dark mode */
    .stNumberInput [data-testid="stNumberInputStepUp"],
    .stNumberInput [data-testid="stNumberInputStepDown"],
    .stNumberInput button {{
        background-color: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    .stNumberInput [data-testid="stNumberInputStepUp"]:hover,
    .stNumberInput [data-testid="stNumberInputStepDown"]:hover,
    .stNumberInput button:hover {{
        background-color: {theme['accent']} !important;
        color: white !important;
        border-color: {theme['accent']} !important;
    }}
    
    .stNumberInput [data-testid="stNumberInputStepUp"] svg,
    .stNumberInput [data-testid="stNumberInputStepDown"] svg,
    .stNumberInput button svg {{
        fill: {theme['text_primary']} !important;
        stroke: {theme['text_primary']} !important;
    }}
    
    .stNumberInput [data-testid="stNumberInputStepUp"]:hover svg,
    .stNumberInput [data-testid="stNumberInputStepDown"]:hover svg,
    .stNumberInput button:hover svg {{
        fill: white !important;
        stroke: white !important;
    }}
    
    /* Number input container */
    .stNumberInput > div {{
        background-color: transparent !important;
    }}
    
    .stNumberInput [data-baseweb="input"] {{
        background-color: {theme['bg_secondary']} !important;
        border-color: {theme['border']} !important;
    }}
    
    /* Select box */
    .stSelectbox > div > div {{
        background: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    /* Info, Success, Warning boxes - use default styling */
    div[data-baseweb="notification"] {{
        background: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']} !important;
    }}
    
    div[data-testid="stMarkdownContainer"] p {{
        color: {theme['text_primary']};
    }}
    
    /* Metric label in sidebar */
    [data-testid="stMetricLabel"] {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stMetricValue"] {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Slider */
    .stSlider > div > div > div > div {{
        background: {theme['accent']} !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
        background: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
        border-radius: 10px;
        border: 1px solid {theme['border']};
    }}
    
    /* Expander content */
    .streamlit-expanderContent {{
        background: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 5px;
        flex-wrap: wrap;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
        background: {theme['bg_secondary']};
        color: {theme['text_primary']} !important;
        border-radius: 10px;
        padding: 8px 15px;
        font-size: var(--font-size-sm);
        border: 1px solid {theme['border']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {theme['accent']} !important;
        color: white !important;
    }}
    
    /* Data table */
    .stDataFrame {{
        border-radius: 15px;
        overflow: hidden;
        font-size: var(--font-size-sm);
    }}
    
    .stDataFrame [data-testid="stDataFrameResizable"] {{
        background: {theme['bg_secondary']};
    }}
    
    /* Column headers and cells */
    .stDataFrame th {{
        background: {theme['bg_card']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    .stDataFrame td {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Streamlit slider */
    .stSlider p, .stSlider label {{
        color: {theme['text_primary']} !important;
    }}
    
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {{
        color: {theme['text_secondary']} !important;
    }}
    
    /* Streamlit info/warning/success/error boxes */
    [data-testid="stNotificationContentInfo"],
    [data-testid="stNotificationContentWarning"],
    [data-testid="stNotificationContentSuccess"],
    [data-testid="stNotificationContentError"] {{
        color: inherit !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        color: {theme['text_primary']} !important;
        background: {theme['bg_secondary']} !important;
    }}
    
    /* Widget labels */
    [data-testid="stWidgetLabel"] {{
        color: {theme['text_primary']} !important;
    }}
    
    /* ============================================
       COMPREHENSIVE LIGHT MODE SUPPORT
       ============================================ */
    
    /* Text Input - complete styling */
    [data-testid="stSidebar"] .stTextInput input {{
        background-color: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border']} !important;
    }}
    
    [data-testid="stSidebar"] .stTextInput input::placeholder {{
        color: {theme['text_secondary']} !important;
    }}
    
    [data-testid="stSidebar"] .stTextInput label {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Expander text inputs */
    .stExpander .stTextInput input {{
        background-color: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border']} !important;
    }}
    
    .stExpander .stTextInput label {{
        color: {theme['text_primary']} !important;
        font-weight: 500;
    }}
    
    /* Expander comprehensive styling for both themes */
    [data-testid="stExpander"] {{
        background-color: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 12px !important;
        overflow: hidden;
    }}
    
    [data-testid="stExpander"] details {{
        background-color: {theme['bg_secondary']} !important;
    }}
    
    [data-testid="stExpander"] summary {{
        background-color: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
        padding: 12px 16px !important;
        border-radius: 12px !important;
    }}
    
    [data-testid="stExpander"] summary:hover {{
        background-color: {theme['bg_card']} !important;
        opacity: 0.95;
    }}
    
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {{
        color: {theme['text_primary']} !important;
        font-weight: 600;
    }}
    
    [data-testid="stExpanderDetails"] {{
        background-color: {theme['bg_secondary']} !important;
        padding: 16px !important;
        border-top: 1px solid {theme['border']} !important;
    }}
    
    /* Expander icon color */
    [data-testid="stExpander"] [data-testid="stIconMaterial"] {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Number Input - complete styling */
    [data-testid="stSidebar"] .stNumberInput input {{
        background-color: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border']} !important;
    }}
    
    /* Number Input + and - buttons in sidebar */
    [data-testid="stSidebar"] .stNumberInput [data-testid="stNumberInputStepUp"],
    [data-testid="stSidebar"] .stNumberInput [data-testid="stNumberInputStepDown"],
    [data-testid="stSidebar"] .stNumberInput button {{
        background-color: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stSidebar"] .stNumberInput [data-testid="stNumberInputStepUp"]:hover,
    [data-testid="stSidebar"] .stNumberInput [data-testid="stNumberInputStepDown"]:hover,
    [data-testid="stSidebar"] .stNumberInput button:hover {{
        background-color: {theme['accent']} !important;
        color: white !important;
        border-color: {theme['accent']} !important;
    }}
    
    [data-testid="stSidebar"] .stNumberInput [data-testid="stNumberInputStepUp"] svg,
    [data-testid="stSidebar"] .stNumberInput [data-testid="stNumberInputStepDown"] svg,
    [data-testid="stSidebar"] .stNumberInput button svg {{
        fill: {theme['text_primary']} !important;
        stroke: {theme['text_primary']} !important;
    }}
    
    [data-testid="stSidebar"] .stNumberInput [data-testid="stNumberInputStepUp"]:hover svg,
    [data-testid="stSidebar"] .stNumberInput [data-testid="stNumberInputStepDown"]:hover svg,
    [data-testid="stSidebar"] .stNumberInput button:hover svg {{
        fill: white !important;
        stroke: white !important;
    }}
    
    /* All input containers */
    [data-testid="stSidebar"] [data-baseweb="input"] {{
        background-color: {theme['bg_secondary']} !important;
    }}
    
    [data-testid="stSidebar"] [data-baseweb="base-input"] {{
        background-color: {theme['bg_secondary']} !important;
        border-color: {theme['border']} !important;
    }}
    
    /* Tabs in sidebar */
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent !important;
    }}
    
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {{
        background-color: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border']} !important;
    }}
    
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: {theme['accent']} !important;
        color: white !important;
    }}
    
    /* ============================================
       FILE UPLOADER STYLING (COMPREHENSIVE)
       ============================================ */
    
    /* File uploader container */
    [data-testid="stFileUploader"] {{
        background-color: {theme['bg_secondary']} !important;
        border: 2px dashed {theme['border']} !important;
        border-radius: 12px !important;
        padding: 16px !important;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: {theme['accent']} !important;
        background-color: {theme['bg_card']} !important;
    }}
    
    /* File uploader label */
    [data-testid="stFileUploader"] label {{
        color: {theme['text_primary']} !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }}
    
    /* File uploader section wrapper */
    [data-testid="stFileUploader"] section {{
        background-color: transparent !important;
    }}
    
    /* File uploader text - all variations */
    [data-testid="stFileUploader"] span {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stFileUploader"] p {{
        color: {theme['text_secondary']} !important;
    }}
    
    [data-testid="stFileUploader"] small {{
        color: {theme['text_secondary']} !important;
        opacity: 0.8;
    }}
    
    /* Drag and drop instruction text */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {{
        color: {theme['text_primary']} !important;
    }}
    
    /* File limit text */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileLimit"] {{
        color: {theme['text_secondary']} !important;
        font-size: 0.75rem !important;
    }}
    
    /* Browse files button */
    [data-testid="stFileUploader"] button {{
        background-color: {theme['accent']} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease;
    }}
    
    [data-testid="stFileUploader"] button:hover {{
        opacity: 0.9 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }}
    
    /* File uploader icons */
    [data-testid="stFileUploader"] svg {{
        fill: {theme['text_primary']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    /* Upload icon specifically */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] svg {{
        fill: {theme['text_secondary']} !important;
        opacity: 0.6;
    }}
    
    /* Uploaded file display */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{
        background-color: {theme['bg_card']} !important;
        border: 1px solid {theme['border']} !important;
        color: {theme['text_primary']} !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }}
    
    /* File name text */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {{
        color: {theme['text_primary']} !important;
        font-weight: 500 !important;
    }}
    
    /* File size text */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"] {{
        color: {theme['text_secondary']} !important;
    }}
    
    /* File uploader delete button */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] {{
        color: {theme['danger']} !important;
    }}
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"]:hover {{
        opacity: 0.8 !important;
    }}
    
    /* Sidebar file uploader specific */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {{
        background-color: {theme['bg_secondary']} !important;
        border-color: {theme['border']} !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {{
        background-color: {theme['accent']} !important;
        color: white !important;
    }}
    
    /* Button in sidebar */
    [data-testid="stSidebar"] .stButton button {{
        background-color: {theme['accent']} !important;
        color: white !important;
        border: none !important;
    }}
    
    [data-testid="stSidebar"] .stButton button:hover {{
        opacity: 0.85 !important;
    }}
    
    /* Secondary button styling */
    [data-testid="stSidebar"] .stButton button[kind="secondary"] {{
        background-color: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border']} !important;
    }}
    
    /* Markdown text in sidebar */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stSidebar"] strong {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stSidebar"] code {{
        background-color: {theme['bg_secondary']} !important;
        color: {theme['accent']} !important;
        border: 1px solid {theme['border']} !important;
    }}
    
    /* Horizontal rule in sidebar */
    [data-testid="stSidebar"] hr {{
        border-color: {theme['border']} !important;
    }}
    
    /* Info box in sidebar */
    [data-testid="stSidebar"] [data-testid="stAlert"] {{
        background-color: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']} !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stAlert"] p {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Slider in sidebar */
    [data-testid="stSidebar"] .stSlider label {{
        color: {theme['text_primary']} !important;
    }}
    
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {{
        color: {theme['text_secondary']} !important;
    }}
    
    /* Slider thumb and track */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {{
        background-color: {theme['accent']} !important;
    }}
    
    [data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Success message in sidebar */
    [data-testid="stSidebar"] .stSuccess {{
        background-color: rgba(26, 127, 55, 0.1) !important;
        color: {theme['success']} !important;
    }}
    
    /* Main content area text */
    .main .block-container {{
        color: {theme['text_primary']};
    }}
    
    /* DataFrame / Table styling */
    [data-testid="stDataFrame"] {{
        background-color: {theme['bg_secondary']} !important;
    }}
    
    [data-testid="stDataFrame"] * {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Info message on main page */
    .main [data-testid="stAlert"] {{
        background-color: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    /* WiFi Config Card */
    .wifi-config-card {{
        background: {theme['bg_card']};
        border-radius: 20px;
        padding: 20px;
        border: 2px solid {theme['accent']};
        margin: 15px 0;
    }}
    
    .wifi-config-title {{
        font-family: 'Poppins', sans-serif;
        font-size: var(--font-size-lg);
        color: {theme['accent']};
        margin-bottom: 15px;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {theme['bg_primary']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {theme['accent']};
        border-radius: 10px;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 30px 20px;
        font-family: 'Poppins', sans-serif;
        color: {theme['text_secondary']};
        border-top: 2px solid {theme['border']};
        margin-top: 40px;
        background: {theme['bg_secondary']};
        border-radius: 15px;
    }}
    
    .footer a {{
        color: {theme['accent']};
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    
    .footer a:hover {{
        opacity: 0.8;
        text-decoration: underline;
    }}
    
    .footer p {{
        margin: 8px 0;
        line-height: 1.6;
    }}
    
    /* Equal height columns for metric cards */
    [data-testid="stHorizontalBlock"] {{
        align-items: stretch !important;
        display: flex !important;
    }}
    
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        display: flex !important;
        flex-direction: column !important;
    }}
    
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {{
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }}
    
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div > div {{
        flex: 1 !important;
    }}
    
    /* ================================================
       RESPONSIVE BREAKPOINTS
       ================================================ */
    
    /* Large Desktop (1400px+) */
    @media (min-width: 1400px) {{
        :root {{
            --metric-value-size: 2rem;
            --metric-icon-size: 2.2rem;
            --title-size: 2.5rem;
            --section-title-size: 1.5rem;
            --card-padding: 20px;
            --gauge-height: 300px;
            --chart-height: 700px;
        }}
        
        [data-testid="stHorizontalBlock"] {{
            gap: 18px !important;
        }}
        
        .metric-card {{
            height: 185px;
        }}
        
        .metric-icon svg {{
            width: 36px;
            height: 36px;
        }}
        
        .wifi-dbm-value {{
            font-size: 1.6rem;
        }}
    }}
    
    /* Desktop (1200px - 1399px) */
    @media (min-width: 1200px) and (max-width: 1399px) {{
        :root {{
            --metric-value-size: 1.75rem;
            --metric-icon-size: 1.8rem;
            --title-size: 2.2rem;
            --section-title-size: 1.4rem;
            --card-padding: 16px;
            --gauge-height: 260px;
            --chart-height: 600px;
        }}
        
        [data-testid="stHorizontalBlock"] {{
            gap: 16px !important;
        }}
        
        .metric-card {{
            height: 175px;
        }}
        
        .metric-icon svg {{
            width: 30px;
            height: 30px;
        }}
        
        .metric-label {{
            font-size: 0.65rem;
            letter-spacing: 1px;
        }}
        
        .wifi-dbm-value {{
            font-size: 1.4rem;
        }}
        
        .wifi-quality-badge {{
            padding: 3px 10px;
            font-size: 0.6rem;
        }}
    }}
    
    /* Small Desktop / Large Tablet (992px - 1199px) */
    @media (min-width: 992px) and (max-width: 1199px) {{
        :root {{
            --metric-value-size: 1.5rem;
            --metric-icon-size: 1.6rem;
            --title-size: 2rem;
            --section-title-size: 1.3rem;
            --card-padding: 12px;
            --card-radius: 14px;
            --gauge-height: 240px;
            --chart-height: 550px;
        }}
        
        [data-testid="stHorizontalBlock"] {{
            gap: 14px !important;
        }}
        
        .metric-card {{
            height: 165px;
        }}
        
        .metric-icon svg {{
            width: 26px;
            height: 26px;
        }}
        
        .metric-label {{
            letter-spacing: 0.8px;
            font-size: 0.6rem;
        }}
        
        .wifi-dbm-value {{
            font-size: 1.25rem;
        }}
        
        .wifi-signal-bar {{
            height: 6px;
        }}
        
        .wifi-quality-badge {{
            padding: 3px 8px;
            font-size: 0.55rem;
        }}
        
        .ml-confidence {{
            font-size: 0.65rem;
        }}
        
        .metric-spacer {{
            height: 26px;
        }}
        
        .gauge-card {{
            min-height: 240px;
            padding: 10px;
        }}
    }}
    
    /* Tablet (768px - 991px) */
    @media (min-width: 768px) and (max-width: 991px) {{
        :root {{
            --metric-value-size: 1.35rem;
            --metric-icon-size: 1.4rem;
            --title-size: 1.7rem;
            --section-title-size: 1.1rem;
            --card-padding: 10px;
            --card-radius: 12px;
            --gauge-height: 200px;
            --chart-height: 450px;
        }}
        
        /* 3-column layout for tablets - 5 cards become 3+2 */
        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
            gap: 8px !important;
        }}
        
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            flex: 1 1 calc(33.33% - 8px) !important;
            min-width: calc(33.33% - 8px) !important;
            max-width: calc(33.33% - 8px) !important;
        }}
        
        /* Last 2 items in 5-col layout take half width each */
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(5)) > [data-testid="stColumn"]:nth-child(4),
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(5)) > [data-testid="stColumn"]:nth-child(5) {{
            flex: 1 1 calc(50% - 8px) !important;
            min-width: calc(50% - 8px) !important;
            max-width: calc(50% - 8px) !important;
        }}
        
        .main-title {{
            letter-spacing: 1px;
            margin-bottom: 15px;
            text-align: center;
        }}
        
        .metric-card {{
            height: auto !important;
            min-height: 140px;
        }}
        
        .metric-icon svg {{
            width: 22px;
            height: 22px;
        }}
        
        .metric-icon {{
            height: 32px;
            margin-bottom: 6px;
        }}
        
        .metric-label {{
            letter-spacing: 0.5px;
            font-size: 0.55rem;
        }}
        
        .section-title {{
            margin: 20px 0 15px 0;
        }}
        
        .stButton > button {{
            padding: 10px 20px;
        }}
        
        .wifi-dbm-value {{
            font-size: 1.1rem;
        }}
        
        .wifi-signal-bar {{
            height: 5px;
            margin: 5px 0;
        }}
        
        .wifi-quality-badge {{
            padding: 2px 6px;
            font-size: 0.5rem;
            margin-top: 4px;
        }}
        
        .ml-confidence {{
            font-size: 0.6rem;
        }}
        
        .metric-spacer {{
            height: 22px;
        }}
        
        .gauge-card {{
            min-height: 220px;
            padding: 8px;
        }}
        
        /* File uploader mobile adjustments */
        [data-testid="stFileUploader"] {{
            padding: 12px !important;
        }}
        
        [data-testid="stFileUploader"] button {{
            padding: 6px 12px !important;
            font-size: 0.8rem !important;
        }}
    }}
    
    /* Large Phone (576px - 767px) */
    @media (min-width: 576px) and (max-width: 767px) {{
        :root {{
            --metric-value-size: 1.3rem;
            --metric-icon-size: 1.2rem;
            --title-size: 1.4rem;
            --section-title-size: 1rem;
            --card-padding: 8px;
            --card-radius: 10px;
            --gauge-height: 180px;
            --chart-height: 400px;
        }}
        
        /* Force 2 columns layout on large phone */
        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
            gap: 8px !important;
        }}
        
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            flex: 1 1 calc(50% - 8px) !important;
            min-width: calc(50% - 8px) !important;
            max-width: calc(50% - 8px) !important;
        }}
        
        /* Last item in 5-col and 3-col layouts takes full width */
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(5)) > [data-testid="stColumn"]:nth-child(5),
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3):last-child) > [data-testid="stColumn"]:nth-child(3) {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }}
        
        .main-title {{
            letter-spacing: 1px;
            margin-bottom: 12px;
            font-size: 1.2rem !important;
            text-align: center;
        }}
        
        .metric-card {{
            height: auto !important;
            min-height: 120px;
            margin: 4px 0;
        }}
        
        /* File uploader large phone responsive */
        [data-testid="stFileUploader"] {{
            padding: 13px !important;
        }}
        
        [data-testid="stFileUploader"] button {{
            padding: 7px 13px !important;
            font-size: 0.82rem !important;
        }}
        
        .metric-icon svg {{
            width: 20px;
            height: 20px;
        }}
        
        .metric-icon {{
            height: 28px;
            margin-bottom: 4px;
        }}
        
        .metric-label {{
            letter-spacing: 0.5px;
            font-size: 0.5rem;
        }}
        
        .metric-value {{
            margin: 4px 0;
            font-size: 1.2rem !important;
        }}
        
        .section-title {{
            margin: 15px 0 10px 0;
            font-size: 1.1rem !important;
        }}
        
        .alert-danger, .alert-warning, .alert-success {{
            padding: 10px 12px;
            font-size: var(--font-size-sm);
        }}
        
        .stButton > button {{
            padding: 8px 12px;
            font-size: var(--font-size-xs);
        }}
        
        .footer {{
            padding: 15px 10px;
        }}
        
        .footer p {{
            font-size: var(--font-size-sm);
        }}
        
        .wifi-dbm-value {{
            font-size: 1rem;
        }}
        
        .wifi-signal-bar {{
            height: 4px;
            margin: 4px 0;
        }}
        
        .wifi-quality-badge {{
            display: none;
        }}
        
        .ml-confidence {{
            font-size: 0.55rem;
        }}
        
        .metric-spacer {{
            height: 18px;
        }}
        
        .gauge-card {{
            min-height: 200px;
            padding: 6px;
        }}
        
        /* File uploader tablet responsive */
        [data-testid="stFileUploader"] {{
            padding: 12px !important;
        }}
        
        [data-testid="stFileUploader"] button {{
            padding: 7px 12px !important;
            font-size: 0.8rem !important;
        }}
    }}
    
    /* Small Phone (max-width: 575px) */
    @media (max-width: 575px) {{
        :root {{
            --metric-value-size: 1.2rem;
            --metric-icon-size: 1rem;
            --title-size: 1.1rem;
            --section-title-size: 0.85rem;
            --card-padding: 6px;
            --card-radius: 8px;
            --gauge-height: 160px;
            --chart-height: 350px;
        }}
        
        /* Force columns to wrap on mobile */
        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
            gap: 10px !important;
            margin-bottom: 12px !important;
        }}
        
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            flex: 1 1 calc(50% - 10px) !important;
            min-width: calc(50% - 10px) !important;
            max-width: calc(50% - 10px) !important;
        }}
        
        /* For 5-column layouts, make 2-2-1 pattern */
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(5)) > [data-testid="stColumn"]:nth-child(5) {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }}
        
        /* For 3-column layouts, stack vertically */
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3):last-child) > [data-testid="stColumn"] {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }}
        
        .main-title {{
            letter-spacing: 0.5px;
            margin-bottom: 16px;
            font-size: 1rem !important;
            padding: 0 5px;
            text-align: center;
        }}
        
        .metric-card {{
            height: auto !important;
            min-height: 100px;
            margin: 6px 0 !important;
            padding: 12px 8px;
        }}
        
        .metric-label {{
            letter-spacing: 0.3px;
            font-size: 0.55rem;
            margin-bottom: 2px;
        }}
        
        .metric-value {{
            margin: 3px 0;
            font-size: 1.2rem !important;
            word-break: break-word;
        }}
        
        .metric-icon {{
            margin-bottom: 4px;
            font-size: 1rem;
        }}
        
        .section-title {{
            margin: 20px 0 12px 0;
            font-size: 0.9rem !important;
            letter-spacing: 0.5px;
            text-align: center;
        }}
        
        .alert-danger, .alert-warning, .alert-success {{
            padding: 8px 10px;
            font-size: 0.7rem;
            border-radius: 8px;
            margin: 4px 0;
        }}
        
        .stButton > button {{
            padding: 6px 10px;
            font-size: 0.65rem;
            letter-spacing: 0.5px;
            border-radius: 16px;
            width: 100%;
        }}
        
        .status-online, .status-offline {{
            padding: 4px 10px;
            font-size: 0.65rem;
            letter-spacing: 0.5px;
        }}
        
        .footer {{
            padding: 10px 6px;
        }}
        
        .footer p {{
            font-size: 0.65rem;
        }}
        
        /* Sidebar adjustments for mobile */
        [data-testid="stSidebar"] {{
            min-width: 240px !important;
            max-width: 85vw !important;
        }}
        
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
            padding: 0.5rem !important;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 5px 8px;
            font-size: 0.65rem;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            flex-wrap: wrap;
            gap: 2px;
        }}
        
        .wifi-dbm-value {{
            font-size: 0.85rem;
        }}
        
        .wifi-signal-bar {{
            height: 3px;
            margin: 2px 0;
        }}
        
        .wifi-quality-badge {{
            display: none;
        }}
        
        .wifi-signal-container {{
            padding: 0 3px;
        }}
        
        .ml-confidence {{
            font-size: 0.45rem;
            display: block;
        }}
        
        .metric-spacer {{
            height: 8px;
        }}
        
        .gauge-card {{
            min-height: 160px;
            padding: 8px;
            margin: 8px 0;
        }}
        
        /* Plotly charts mobile fix */
        .stPlotlyChart {{
            overflow-x: auto !important;
        }}
        
        .js-plotly-plot .plotly .modebar {{
            display: none !important;
        }}
        
        /* Main content spacing */
        [data-testid="stVerticalBlock"] > div {{
            margin-bottom: 8px !important;
        }}
        
        /* Block container spacing */
        .block-container {{
            padding: 1rem 0.8rem !important;
        }}
        
        /* File uploader mobile responsive */
        [data-testid="stFileUploader"] {{
            padding: 10px !important;
            font-size: 0.75rem !important;
        }}
        
        [data-testid="stFileUploader"] button {{
            padding: 6px 10px !important;
            font-size: 0.75rem !important;
        }}
        
        [data-testid="stFileUploader"] label {{
            font-size: 0.8rem !important;
        }}
    }}
    
    /* Extra Small Phone (max-width: 400px) */
    @media (max-width: 400px) {{
        /* Stack all columns vertically */
        [data-testid="stHorizontalBlock"] {{
            gap: 10px !important;
            margin-bottom: 14px !important;
        }}
        
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }}
        
        .main-title {{
            font-size: 0.9rem !important;
            padding: 0 3px;
            margin-bottom: 16px;
        }}
        
        .metric-value {{
            font-size: 1.3rem !important;
        }}
        
        .metric-icon {{
            font-size: 0.9rem;
        }}
        
        .metric-icon svg {{
            width: 24px !important;
            height: 24px !important;
        }}
        
        .metric-label {{
            font-size: 0.5rem;
        }}
        
        .metric-card {{
            height: auto !important;
            min-height: 90px;
            padding: 10px 8px;
            margin: 6px 0 !important;
        }}
        
        .wifi-dbm-value {{
            font-size: 0.8rem;
        }}
        
        .wifi-signal-bar {{
            height: 3px;
        }}
        
        .gauge-card {{
            min-height: 140px;
            margin: 8px 0 !important;
        }}
        
        .section-title {{
            font-size: 0.85rem !important;
            margin: 18px 0 10px 0;
        }}
        
        /* Hide less important elements on very small screens */
        .ml-confidence {{
            display: none;
        }}
    }}
    
    /* Touch-friendly improvements */
    @media (hover: none) and (pointer: coarse) {{
        .metric-card {{
            transition: none;
        }}
        
        .metric-card:hover {{
            transform: none;
        }}
        
        .stButton > button {{
            min-height: 44px;
        }}
        
        [data-testid="stSidebar"] button {{
            min-height: 40px;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ==================== MQTT CONFIGURATION ====================
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
TOPIC_DATA = "projek/asma/data_sensor"
TOPIC_COMMAND = "projek/asma/command"  # For sending commands/ML results to ESP32
TOPIC_RESPONSE = "projek/asma/response"  # For receiving responses from ESP32

# Initialize session state
if 'mqtt_connected' not in st.session_state:
    st.session_state.mqtt_connected = False
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {
        'suhu': 0,
        'lembab': 0,
        'adc_raw': 0,
        'adc_percent': 0,
        'local_aqi': 0,
        'aqi_category': 'N/A',
        'baseline_adc': 500,
        'quality': 'N/A',
        'level': 'safe',
        'ip': 'N/A',
        'rssi': 0,
        'uptime': 0,
        'weather_temp': None,
        'weather_aqi': None,
        'weather_aqi_cat': None
    }
if 'data_history' not in st.session_state:
    st.session_state.data_history = {
        'time': deque(maxlen=100),
        'suhu': deque(maxlen=100),
        'lembab': deque(maxlen=100),
        'aqi': deque(maxlen=100)
    }
if 'last_update' not in st.session_state:
    st.session_state.last_update = "Never"
if 'client' not in st.session_state:
    st.session_state.client = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
# MQTT topic storage (separate from widget keys)
if 'mqtt_topic_data' not in st.session_state:
    st.session_state.mqtt_topic_data = TOPIC_DATA
# ML prediction state
if 'ml_prediction' not in st.session_state:
    st.session_state.ml_prediction = {'status': 'N/A', 'confidence': None}
# MQTT worker queue for thread-safe communication
if 'mqtt_in_q' not in st.session_state:
    st.session_state.mqtt_in_q = queue.Queue()
if 'mqtt_worker_started' not in st.session_state:
    st.session_state.mqtt_worker_started = False
# Device response queue for calibration/command feedback
if 'device_response_q' not in st.session_state:
    st.session_state.device_response_q = queue.Queue()
if 'last_device_response' not in st.session_state:
    st.session_state.last_device_response = None
# Current city setting
if 'current_city' not in st.session_state:
    st.session_state.current_city = "Cilegon"

# ==================== MQTT WORKER (Background Thread) ====================
def mqtt_worker(broker, port, topic_sensor, topic_response, in_q, response_q):
    """Background worker thread for MQTT - runs continuously"""
    try:
        client = mqtt.Client(
            client_id=f"dashboard_worker_{int(time.time())}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            clean_session=True
        )
    except (AttributeError, TypeError):
        client = mqtt.Client(
            client_id=f"dashboard_worker_{int(time.time())}",
            clean_session=True
        )
    
    def _on_connect(c, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.mqtt_status("connected", broker, port, f"{topic_sensor}, {topic_response}")
            c.subscribe(topic_sensor, qos=1)
            c.subscribe(topic_response, qos=1)  # Subscribe to response topic
            st.session_state.mqtt_connected = True
        else:
            logger.mqtt_status("error", broker, port, rc=f"Connection failed (rc={rc})")
            st.session_state.mqtt_connected = False
    
    def _on_message(c, userdata, msg, properties=None):
        try:
            data = json.loads(msg.payload.decode())
            wib = timezone(timedelta(hours=7))
            
            # Route message based on topic
            if msg.topic == topic_sensor:
                in_q.put({
                    "ts": datetime.now(wib).strftime("%H:%M:%S"),
                    "payload": data
                })
            elif msg.topic == topic_response:
                # Response from ESP32 (calibration, set_city, etc.)
                response_q.put({
                    "ts": datetime.now(wib).strftime("%H:%M:%S"),
                    "payload": data
                })
                logger.actuator_response(data)
        except Exception as e:
            logger.error(f"Error parsing MQTT message: {e}")
    
    def _on_disconnect(c, userdata, rc, properties=None):
        st.session_state.mqtt_connected = False
        logger.mqtt_status("disconnected", rc=rc)

    client.on_connect = _on_connect
    client.on_message = _on_message
    client.on_disconnect = _on_disconnect

    while True:
        try:
            logger.mqtt_status("connecting", broker, port)
            client.connect(broker, port, keepalive=60)
            client.loop_forever()
        except Exception as e:
            logger.error(f"MQTT Worker error: {e}")
            st.session_state.mqtt_connected = False
            time.sleep(3)  # Wait before reconnecting

# ==================== MQTT PUBLISHER (Cached) ====================
@st.cache_resource
def get_mqtt_publisher():
    """Get a cached MQTT client for publishing"""
    try:
        client = mqtt.Client(
            client_id=f"dashboard_publisher_{int(time.time())}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
    except (AttributeError, TypeError):
        client = mqtt.Client(client_id=f"dashboard_publisher_{int(time.time())}")
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
        logger.success("MQTT Publisher connected")
    except Exception as e:
        logger.error(f"MQTT Publisher connection error: {e}")
    return client

def send_ml_prediction_to_esp32(status, confidence):
    """Send ML prediction result to ESP32 via MQTT"""
    try:
        publisher = get_mqtt_publisher()
        payload = json.dumps({
            "cmd": "set_status",
            "status": status,
            "confidence": int(confidence) if confidence else 0
        })
        publisher.publish(TOPIC_COMMAND, payload)
        logger.command_sent("set_status", status=status, confidence=f"{confidence}%" if confidence else "N/A")
        return True
    except Exception as e:
        logger.error(f"Error sending ML prediction: {e}")
    return False

def send_command_to_esp32(command, **kwargs):
    """Send command to ESP32 via MQTT"""
    try:
        publisher = get_mqtt_publisher()
        payload_dict = {"cmd": command}
        payload_dict.update(kwargs)
        payload = json.dumps(payload_dict)
        publisher.publish(TOPIC_COMMAND, payload)
        logger.command_sent(command, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Error sending command: {e}")
    return False

def process_device_responses():
    """Process responses from ESP32"""
    q = st.session_state.device_response_q
    responses = []
    while not q.empty():
        try:
            item = q.get_nowait()
            responses.append(item)
            st.session_state.last_device_response = item
        except queue.Empty:
            break
    return responses

# Start MQTT worker thread (only once)
if not st.session_state.mqtt_worker_started:
    logger.info("🚀 Starting MQTT Worker thread...")
    worker_thread = threading.Thread(
        target=mqtt_worker,
        args=(MQTT_BROKER, MQTT_PORT, TOPIC_DATA, TOPIC_RESPONSE, 
              st.session_state.mqtt_in_q, st.session_state.device_response_q),
        daemon=True
    )
    worker_thread.start()
    st.session_state.mqtt_worker_started = True
    time.sleep(1)  # Give worker time to connect

# ==================== PROCESS INCOMING DATA ====================
def extract_sensor_values(data):
    """Extract sensor values from flat JSON format (AQI-based from stage4.ino)"""
    # New AQI-based format from updated stage4.ino
    suhu = data.get('suhu', 0)
    lembab = data.get('lembab', 0)
    adc_raw = data.get('adc_raw', 0)
    adc_percent = data.get('adc_percent', 0)
    local_aqi = data.get('local_aqi', 0)
    aqi_category = data.get('aqi_category', 'N/A')
    
    # Fallback for old nested format (backward compatibility)
    if 'environment' in data:
        env = data.get('environment', {})
        suhu = env.get('temp', suhu)
        lembab = env.get('humid', lembab)
    
    return suhu, lembab, adc_raw, local_aqi, aqi_category

def process_incoming_data():
    """Process data from MQTT worker queue"""
    q = st.session_state.mqtt_in_q
    data_received = False
    
    while not q.empty():
        try:
            item = q.get_nowait()
            data = item["payload"]
            ts = item["ts"]
            
            # Update sensor data (handle nested JSON structure)
            st.session_state.sensor_data = data
            st.session_state.last_update = ts
            
            # Extract data using helper function (handles both formats)
            suhu, lembab, adc_raw, local_aqi, aqi_category = extract_sensor_values(data)
            
            # Add to history (store AQI for visualization)
            wib = timezone(timedelta(hours=7))
            st.session_state.data_history['time'].append(datetime.now(wib))
            st.session_state.data_history['suhu'].append(suhu)
            st.session_state.data_history['lembab'].append(lembab)
            st.session_state.data_history['aqi'].append(local_aqi)
            
            # Log sensor data to terminal
            rssi = data.get('rssi', None)
            logger.sensor_data(suhu, lembab, local_aqi, rssi=rssi, adc_raw=adc_raw)
            
            # ML Prediction using AQI and sensor data (AQI range: 0-100)
            aqi = local_aqi
            
            if ml_model is not None:
                try:
                    X = [[suhu, lembab, aqi]]
                    
                    # Scale input if scaler is available (required for SVM)
                    if ml_scaler is not None:
                        X = ml_scaler.transform(X)
                    
                    pred_label = ml_model.predict(X)[0]
                    
                    # Decode label using label encoder if available
                    if ml_label_encoder is not None:
                        pred_text = ml_label_encoder.inverse_transform([pred_label])[0]
                        # Map to display text
                        text_map = {"AMAN": "Aman", "HATI-HATI": "Waspada", "BAHAYA": "Bahaya"}
                        pred_text = text_map.get(pred_text, pred_text)
                    else:
                        # Fallback to manual mapping
                        label_map = {0: "Aman", 1: "Waspada", 2: "Bahaya"}
                        pred_text = label_map.get(pred_label, "Unknown")
                    
                    confidence = None
                    
                    if hasattr(ml_model, 'predict_proba'):
                        # Use scaled X for probability calculation
                        proba = ml_model.predict_proba(X)[0]
                        confidence = float(np.max(proba)) * 100
                    
                    st.session_state.ml_prediction = {'status': pred_text, 'confidence': confidence}
                    
                    # Log ML prediction to terminal
                    sent_ok = send_ml_prediction_to_esp32(pred_text, confidence)
                    logger.ml_prediction(pred_text, confidence, suhu=suhu, lembab=lembab, aqi=aqi, sent_ok=sent_ok)
                    
                except Exception as e:
                    logger.error(f"ML Prediction error: {e}")
                    st.session_state.ml_prediction = {'status': 'Error', 'confidence': None}
            
            # Check alerts
            check_alerts(data)
            data_received = True
            
            # Mark as connected since we received data
            st.session_state.mqtt_connected = True
            
        except queue.Empty:
            break
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    # Check if connection is still alive based on last update
    if st.session_state.last_update != "Never":
        try:
            # Parse last update time
            last_time = datetime.strptime(st.session_state.last_update, "%H:%M:%S")
            wib = timezone(timedelta(hours=7))
            now = datetime.now(wib)
            last_time = last_time.replace(year=now.year, month=now.month, day=now.day, tzinfo=wib)
            
            # If no data for more than 30 seconds, mark as disconnected
            time_diff = (now - last_time).total_seconds()
            if time_diff > 30:
                st.session_state.mqtt_connected = False
            elif data_received:
                # Only set to True if we just received data
                st.session_state.mqtt_connected = True
            # Otherwise keep current state
        except Exception as e:
            logger.warning(f"Error checking timeout: {e}")
            # If we received data, mark as connected
            if data_received:
                st.session_state.mqtt_connected = True
    
    return data_received

# MQTT callbacks (legacy - kept for manual connect option)
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback when connected to MQTT broker"""
    rc_codes = {
        0: "Connected successfully",
        1: "Incorrect protocol version",
        2: "Invalid client identifier",
        3: "Server unavailable",
        4: "Bad username or password",
        5: "Not authorized"
    }
    if rc == 0:
        st.session_state.mqtt_connected = True
        # Subscribe to data topic
        client.subscribe(st.session_state.mqtt_topic_data, qos=1)
    else:
        st.session_state.mqtt_connected = False
        logger.error(f"Connection failed: {rc_codes.get(rc, f'Unknown error code {rc}')}")

def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        data = json.loads(msg.payload.decode())
        
        if topic == st.session_state.mqtt_topic_data:
            st.session_state.sensor_data = data
            wib = timezone(timedelta(hours=7))
            st.session_state.last_update = datetime.now(wib).strftime("%H:%M:%S")
            
            # Add to history - use new AQI-based format
            st.session_state.data_history['time'].append(datetime.now(wib))
            st.session_state.data_history['suhu'].append(data.get('suhu', 0))
            st.session_state.data_history['lembab'].append(data.get('lembab', 0))
            st.session_state.data_history['aqi'].append(data.get('local_aqi', 0))
            
            # ML Prediction - use local_aqi for prediction
            suhu = data.get('suhu', 0)
            lembab = data.get('lembab', 0)
            aqi = data.get('local_aqi', 0)
            
            if ml_model is not None:
                try:
                    X = [[suhu, lembab, aqi]]
                    
                    # Scale input if scaler is available (required for SVM)
                    if ml_scaler is not None:
                        X = ml_scaler.transform(X)
                    
                    pred_label = ml_model.predict(X)[0]
                    
                    # Decode label using label encoder if available
                    if ml_label_encoder is not None:
                        pred_text = ml_label_encoder.inverse_transform([pred_label])[0]
                        # Map to display text
                        text_map = {"AMAN": "Aman", "HATI-HATI": "Waspada", "BAHAYA": "Bahaya"}
                        pred_text = text_map.get(pred_text, pred_text)
                    else:
                        # Fallback to manual mapping
                        label_map = {0: "Aman", 1: "Waspada", 2: "Bahaya"}
                        pred_text = label_map.get(pred_label, "Unknown")
                    
                    confidence = None
                    
                    # Get confidence if available
                    if hasattr(ml_model, 'predict_proba'):
                        proba = ml_model.predict_proba(X)[0]
                        confidence = float(np.max(proba)) * 100
                        
                        # Publish prediction to MQTT
                        prediction_payload = {
                            "timestamp": datetime.now(wib).isoformat(),
                            "sensor_data": {
                                "temperature": float(suhu),
                                "humidity": float(lembab),
                                "local_aqi": int(aqi)
                            },
                            "prediction": {
                                "status": pred_text.upper(),
                                "class": int(pred_label),
                                "confidence": round(confidence, 2),
                                "probabilities": {
                                    "aman": round(float(proba[0]) * 100, 2) if len(proba) > 0 else 0,
                                    "waspada": round(float(proba[1]) * 100, 2) if len(proba) > 1 else 0,
                                    "bahaya": round(float(proba[2]) * 100, 2) if len(proba) > 2 else 0
                                }
                            }
                        }
                        publish_prediction(prediction_payload)
                    
                    st.session_state.ml_prediction = {'status': pred_text, 'confidence': confidence}
                except Exception as e:
                    st.session_state.ml_prediction = {'status': 'Error', 'confidence': None}
            else:
                st.session_state.ml_prediction = {'status': 'N/A', 'confidence': None}
            
            # Check alerts
            check_alerts(data)
            
    except Exception as e:
        logger.error(f"Error parsing message: {e}")

def check_alerts(data):
    alerts = []
    
    # Get AQI-based data from ESP32
    local_aqi = data.get('local_aqi', 0)
    aqi_category = data.get('aqi_category', '')
    suhu = data.get('suhu', 0)
    lembab = data.get('lembab', 0)
    weather_aqi = data.get('weather_aqi', 0)
    
    # Check local AQI status (0-500 scale)
    # AMAN: 0-100, HATI-HATI: 101-200, BAHAYA: >200
    if local_aqi > 200:
        alerts.append(('danger', f'AQI BERBAHAYA! ({local_aqi}) - Udara sangat tidak sehat!'))
    elif local_aqi > 100:
        alerts.append(('warning', f'AQI Tidak Sehat ({local_aqi}) - Perlu dimonitor'))
    
    # Compare with AccuWeather AQI if available
    if weather_aqi and weather_aqi > 100:
        alerts.append(('info', f'AccuWeather AQI: {weather_aqi} (tidak sehat)'))
    
    # Temperature alerts
    if suhu > 35:
        alerts.append(('warning', f'Suhu tinggi terdeteksi! ({suhu:.1f}°C)'))
    
    # Humidity alerts
    if lembab > 80:
        alerts.append(('warning', f'Kelembaban tinggi terdeteksi! ({lembab:.0f}%)'))
    elif lembab < 30:
        alerts.append(('warning', f'Kelembaban rendah terdeteksi! ({lembab:.0f}%)'))
    
    st.session_state.alerts = alerts

def on_disconnect(client, userdata, rc, properties=None):
    """Callback when disconnected from MQTT broker"""
    st.session_state.mqtt_connected = False
    if rc != 0:
        logger.warning(f"Unexpected disconnection (rc={rc}). Will attempt to reconnect...")

def connect_mqtt(broker, port, topic_data):
    """Connect to MQTT broker with improved error handling"""
    try:
        # Disconnect existing client properly
        if st.session_state.client:
            try:
                st.session_state.client.loop_stop()
                st.session_state.client.disconnect()
            except:
                pass
            st.session_state.client = None
            st.session_state.mqtt_connected = False
        
        st.session_state.mqtt_topic_data = topic_data
        
        # Create new client with unique ID - paho-mqtt v2.x compatible
        client_id = f"streamlit_dashboard_{int(time.time())}"
        try:
            # Try paho-mqtt v2.x API
            client = mqtt.Client(
                client_id=client_id,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                clean_session=True
            )
        except (AttributeError, TypeError):
            # Fallback for paho-mqtt v1.x
            client = mqtt.Client(client_id=client_id, clean_session=True)
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        
        # Set connection timeout
        client.connect(broker, port, keepalive=60)
        client.loop_start()
        
        # Wait for connection to establish
        timeout = 5
        start_time = time.time()
        while not st.session_state.mqtt_connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if st.session_state.mqtt_connected:
            st.session_state.client = client
            return True
        else:
            client.loop_stop()
            client.disconnect()
            st.warning("Connection timeout. Please check broker address and try again.")
            return False
            
    except Exception as e:
        st.session_state.mqtt_connected = False
        st.error(f"Connection failed: {str(e)}")
        return False

def publish_prediction(pred_data):
    """Publish ML prediction results to MQTT"""
    try:
        publisher = get_mqtt_publisher()
        payload = json.dumps(pred_data)
        publisher.publish(TOPIC_PREDICTION, payload, qos=1)
        return True
    except Exception as e:
        logger.error(f"Error publishing prediction: {e}")
    return False

# Process incoming data from MQTT worker
process_incoming_data()

# Process device responses (calibration, set_city, etc.)
process_device_responses()

# Show ML model error if any
if ml_model_error:
    st.error(ml_model_error, icon="⚠️")

# Auto refresh
st_autorefresh(interval=2000, key="refresh")

# ==================== CUSTOM HEADER WITH THEME TOGGLE ====================
# Get device info for header
device_ip = st.session_state.sensor_data.get('ip', 'N/A')
uptime = st.session_state.sensor_data.get('uptime', 0)
uptime_str = f"{uptime // 3600}h {(uptime % 3600) // 60}m" if uptime > 0 else "N/A"

# Connection status
status_class = "status-online" if st.session_state.mqtt_connected else "status-offline"
status_text = "ONLINE" if st.session_state.mqtt_connected else "OFFLINE"

# Theme icon
theme_icon = SVG_ICONS['sun'] if st.session_state.theme == 'dark' else SVG_ICONS['moon']
next_theme = 'light' if st.session_state.theme == 'dark' else 'dark'
theme_text = "Light" if st.session_state.theme == 'dark' else "Dark"

# Custom header HTML
st.markdown(f'''
<div class="custom-header">
    <div class="header-left">
        <span class="header-title">{SVG_ICONS["activity"]} Smart Environment Monitor</span>
        <span class="status-badge {status_class}">
            <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="8"/></svg>
            {status_text}
        </span>
        <span class="header-info">📡 {st.session_state.last_update}</span>
    </div>
</div>
''', unsafe_allow_html=True)

# Theme toggle button (using Streamlit button for functionality)
col_spacer, col_theme = st.columns([9, 1])
with col_theme:
    if st.button(f"{'☀️' if st.session_state.theme == 'dark' else '🌙'} {theme_text}", key="theme_toggle", use_container_width=True):
        st.session_state.theme = next_theme
        st.rerun()

# ML Model status in expander (moved from sidebar)
with st.expander("⚙️ Settings", expanded=False):
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        st.markdown("### 🤖 ML Model")
        if ml_model is not None:
            model_name = os.path.basename(st.session_state.custom_model_path) if st.session_state.custom_model_path else MODEL_PATH
            st.success(f"✅ Loaded: `{model_name}`")
        else:
            st.warning("⚠️ No model loaded")
            if ml_model_error:
                st.caption(ml_model_error)
        
        uploaded_file = st.file_uploader(
            "Upload a .pkl model file",
            type=['pkl'],
            key="model_uploader",
            help="Upload a trained scikit-learn model (.pkl file)"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("💾 Load Model", use_container_width=True, disabled=uploaded_file is None):
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    load_ml_model.clear()
                    st.session_state.custom_model_path = temp_path
                    st.success("✅ Model uploaded!")
                    st.rerun()
        with col_btn2:
            if st.button("🔄 Reset Default", use_container_width=True):
                if st.session_state.custom_model_path and os.path.exists(st.session_state.custom_model_path):
                    try:
                        os.unlink(st.session_state.custom_model_path)
                    except:
                        pass
                st.session_state.custom_model_path = None
                load_ml_model.clear()
                st.success("✅ Reset to default model!")
                st.rerun()
    
    with col_set2:
        st.markdown("### 📡 MQTT Settings")
        st.markdown(f"**Broker:** `{MQTT_BROKER}:{MQTT_PORT}`")
        st.markdown(f"**Data Topic:** `{st.session_state.mqtt_topic_data}`")
        if device_ip and device_ip != 'N/A':
            st.markdown(f"**Device IP:** `{device_ip}`")
            st.markdown(f"**Uptime:** {uptime_str}")
        st.success("✅ Auto-connect enabled!")
    
    # Device Control Section
    st.markdown("---")
    st.markdown("### 🎛️ Device Control")
    
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    
    with col_ctrl1:
        st.markdown("**🔧 Kalibrasi Sensor**")
        baseline_adc = st.session_state.sensor_data.get('baseline_adc', 500)
        st.caption(f"Baseline ADC: `{baseline_adc}`")
        
        # CALIBRATE - Same as pressing 'c' in Serial Monitor
        if st.button("📏 Kalibrasi", use_container_width=True, type="primary", help="Kalibrasi sensor MQ135 di udara bersih (sama dengan tekan 'c' di Serial Monitor)"):
            if send_command_to_esp32("calibrate"):
                st.info("⏳ Kalibrasi dimulai... Pastikan sensor di udara bersih!")
            else:
                st.error("❌ Gagal mengirim perintah")
        
        st.caption("☝️ Kalibrasi sensor gas MQ135 di udara bersih")
        
        new_baseline = st.number_input("Baseline Manual", min_value=100, max_value=3000, value=max(100, baseline_adc), key="baseline_input")
        if st.button("💾 Set Baseline", use_container_width=True):
            if send_command_to_esp32("set_baseline", value=new_baseline):
                st.success(f"✅ Baseline: {new_baseline}")
            else:
                st.error("❌ Gagal")
        
        st.markdown("---")
        st.markdown("**🌡️ Offset Suhu**")
        temp_offset = st.number_input("Offset Suhu (°C)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5, key="temp_offset_input", help="Tambahkan ke pembacaan DHT22")
        if st.button("🌡️ Set Temp Offset", use_container_width=True):
            if send_command_to_esp32("set_temp_offset", value=temp_offset):
                st.success(f"✅ Offset suhu: {temp_offset:+.1f}°C")
            else:
                st.error("❌ Gagal")
        
        st.markdown("**💧 Offset Kelembaban**")
        humid_offset = st.number_input("Offset Humid (%)", min_value=-30.0, max_value=30.0, value=0.0, step=1.0, key="humid_offset_input", help="Tambahkan ke pembacaan DHT22")
        if st.button("💧 Set Humid Offset", use_container_width=True):
            if send_command_to_esp32("set_humid_offset", value=humid_offset):
                st.success(f"✅ Offset humid: {humid_offset:+.1f}%")
            else:
                st.error("❌ Gagal")
        
        st.markdown("---")
        col_get, col_reset = st.columns(2)
        with col_get:
            if st.button("📋 Get Calib", use_container_width=True, help="Lihat semua nilai kalibrasi"):
                if send_command_to_esp32("get_calibration"):
                    st.info("⏳ Mengambil...")
        with col_reset:
            if st.button("🔄 Reset All", use_container_width=True, help="Reset semua kalibrasi ke default"):
                if send_command_to_esp32("reset_calibration"):
                    st.warning("⚠️ Kalibrasi direset")
    
    with col_ctrl2:
        st.markdown("**🌍 Lokasi Cuaca**")
        st.caption(f"Kota saat ini: `{st.session_state.current_city}`")
        
        new_city = st.text_input("Nama Kota", value=st.session_state.current_city, key="city_input", placeholder="Contoh: Cilegon, Jakarta, Purwakarta")
        if st.button("📍 Set Kota", use_container_width=True, help="Set kota untuk AccuWeather API"):
            if new_city.strip():
                if send_command_to_esp32("set_city", city=new_city.strip()):
                    st.session_state.current_city = new_city.strip()
                    st.success(f"✅ Kota diset ke: {new_city}")
                else:
                    st.error("❌ Gagal mengirim perintah")
            else:
                st.warning("⚠️ Masukkan nama kota")
        
        if st.button("🌤️ Refresh Cuaca", use_container_width=True, help="Ambil data cuaca terbaru"):
            if send_command_to_esp32("get_weather"):
                st.info("⏳ Mengambil data cuaca...")
            else:
                st.error("❌ Gagal mengirim perintah")
    
    with col_ctrl3:
        st.markdown("**🔊 Kontrol Buzzer**")
        col_bz1, col_bz2 = st.columns(2)
        with col_bz1:
            if st.button("🔔 Buzzer ON", use_container_width=True):
                if send_command_to_esp32("buzzer", on=True):
                    st.success("✅ Buzzer aktif")
        with col_bz2:
            if st.button("🔕 Buzzer OFF", use_container_width=True):
                if send_command_to_esp32("buzzer", on=False):
                    st.success("✅ Buzzer mati")
        
        st.markdown("**📊 Status Perangkat**")
        if st.button("📋 Get Status", use_container_width=True, help="Ambil status sensor terkini"):
            if send_command_to_esp32("get_status"):
                st.info("⏳ Mengambil status...")
            else:
                st.error("❌ Gagal mengirim perintah")
        
        if st.button("🔄 Restart ESP32", use_container_width=True, type="secondary"):
            if send_command_to_esp32("restart"):
                st.warning("⚠️ ESP32 akan restart...")
            else:
                st.error("❌ Gagal mengirim perintah")
    
    # Manual Emoji/Mood Control Section
    st.markdown("---")
    st.markdown("### 😀 Kontrol Emoji Manual")
    st.caption("Pilih emoji untuk ditampilkan di OLED. Akan bertahan **30 detik** lalu kembali ke mode otomatis.")
    
    col_mood1, col_mood2, col_mood3 = st.columns(3)
    
    with col_mood1:
        if st.button("😊 AMAN", use_container_width=True, help="Tampilkan mata senang (30 detik)"):
            if send_command_to_esp32("set_mood", mood="AMAN"):
                st.success("✅ Emoji AMAN aktif (30s)")
            else:
                st.error("❌ Gagal mengirim")
    
    with col_mood2:
        if st.button("😟 HATI-HATI", use_container_width=True, help="Tampilkan mata waspada (30 detik)"):
            if send_command_to_esp32("set_mood", mood="HATI-HATI"):
                st.success("✅ Emoji HATI-HATI aktif (30s)")
            else:
                st.error("❌ Gagal mengirim")
    
    with col_mood3:
        if st.button("❌ BAHAYA", use_container_width=True, help="Tampilkan mata X X (30 detik)"):
            if send_command_to_esp32("set_mood", mood="BAHAYA"):
                st.success("✅ Emoji BAHAYA aktif (30s)")
            else:
                st.error("❌ Gagal mengirim")
    
    # Show last device response
    if st.session_state.last_device_response:
        resp = st.session_state.last_device_response
        st.markdown("---")
        st.markdown("**📥 Response Terakhir dari ESP32:**")
        st.json(resp.get('payload', {}))

# Alerts
if st.session_state.alerts:
    for alert_type, message in st.session_state.alerts:
        alert_icon = SVG_ICONS['alert'].replace('width="48"', 'width="20"').replace('height="48"', 'height="20"')
        st.markdown(f'<div class="alert-{alert_type}">{alert_icon} {message}</div>', unsafe_allow_html=True)

# Metric cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    suhu_val, _, _, _, _ = extract_sensor_values(st.session_state.sensor_data)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{SVG_ICONS['temperature']}</div>
        <div class="metric-label">Temperature</div>
        <div class="metric-value">{suhu_val:.1f}°C</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    _, lembab_val, _, _, _ = extract_sensor_values(st.session_state.sensor_data)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{SVG_ICONS['humidity']}</div>
        <div class="metric-label">Humidity</div>
        <div class="metric-value">{lembab_val:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    _, _, adc_raw, local_aqi, aqi_category = extract_sensor_values(st.session_state.sensor_data)
    weather_aqi = st.session_state.sensor_data.get('weather_aqi', 0)
    
    # AQI-based icons and colors (0-500 scale)
    # AMAN: 0-100, HATI-HATI: 101-200, BAHAYA: >200
    if local_aqi <= 50:
        aqi_icon = SVG_ICONS['gas']
        aqi_color = theme['success']        # Good
    elif local_aqi <= 100:
        aqi_icon = SVG_ICONS['gas']
        aqi_color = theme['success']        # Moderate (still safe)
    elif local_aqi <= 150:
        aqi_icon = SVG_ICONS['alert']
        aqi_color = theme['warning']        # USG
    elif local_aqi <= 200:
        aqi_icon = SVG_ICONS['alert']
        aqi_color = theme['warning']        # Unhealthy
    else:
        aqi_icon = SVG_ICONS['fire']
        aqi_color = theme['danger']         # Very Unhealthy / Hazardous
    
    weather_info = f"AccuWeather: {weather_aqi}" if weather_aqi else aqi_category
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon" style="color: {aqi_color};">{aqi_icon}</div>
        <div class="metric-label">Air Quality Index</div>
        <div class="metric-value" style="color: {aqi_color};">{local_aqi}</div>
        <div class="ml-confidence">{weather_info}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # ML Prediction Card
    ml_pred = st.session_state.ml_prediction
    ml_status = ml_pred.get('status', 'N/A')
    ml_confidence = ml_pred.get('confidence')
    
    # Status colors
    if ml_status == "Aman":
        status_color = theme['success']
        ml_icon = '''<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'''
    elif ml_status == "Waspada":
        status_color = theme['warning']
        ml_icon = SVG_ICONS['alert']
    elif ml_status == "Bahaya":
        status_color = theme['danger']
        ml_icon = SVG_ICONS['fire']
    else:
        status_color = theme['text_secondary']
        ml_icon = '''<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>'''
    
    conf_text = f"({ml_confidence:.1f}%)" if ml_confidence else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon" style="color: {status_color};">{ml_icon}</div>
        <div class="metric-label">ML Prediction</div>
        <div class="metric-value" style="color: {status_color};">{ml_status}</div>
        <div class="ml-confidence">{conf_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Gauges
st.markdown(f'<p class="section-title">{SVG_ICONS["gauge"]} Live Gauges</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

# Get theme colors for charts
chart_text_color = '#e6edf3' if st.session_state.theme == 'dark' else '#24292f'
chart_bg = 'rgba(0,0,0,0)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,0.8)'

# Professional color palette
temp_color = '#f0883e'  # Orange
humid_color = '#58a6ff'  # Blue
aqi_color = '#3fb950'  # Green (default for good AQI)

with col1:
    temp_val, _, _, _, _ = extract_sensor_values(st.session_state.sensor_data)
    fig_temp = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=temp_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Temperature", 'font': {'size': 12, 'color': chart_text_color, 'family': 'Poppins'}},
        number={'font': {'size': 22, 'color': chart_text_color, 'family': 'Poppins'}, 'suffix': '°C'},
        delta={'reference': 25, 'increasing': {'color': "#f85149"}, 'font': {'size': 9}},
        gauge={
            'axis': {'range': [0, 50], 'tickcolor': chart_text_color, 'tickfont': {'color': chart_text_color, 'size': 8}},
            'bar': {'color': temp_color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': chart_text_color,
            'steps': [
                {'range': [0, 20], 'color': 'rgba(88, 166, 255, 0.3)'},
                {'range': [20, 30], 'color': 'rgba(63, 185, 80, 0.3)'},
                {'range': [30, 40], 'color': 'rgba(240, 136, 62, 0.3)'},
                {'range': [40, 50], 'color': 'rgba(248, 81, 73, 0.4)'}
            ],
            'threshold': {
                'line': {'color': "#f85149", 'width': 3},
                'thickness': 0.75,
                'value': 35
            }
        }
    ))
    fig_temp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': chart_text_color, 'family': 'Poppins'},
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        autosize=True
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with col2:
    _, humid_val, _, _, _ = extract_sensor_values(st.session_state.sensor_data)
    fig_humid = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=humid_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Humidity", 'font': {'size': 12, 'color': chart_text_color, 'family': 'Poppins'}},
        number={'font': {'size': 22, 'color': chart_text_color, 'family': 'Poppins'}, 'suffix': '%'},
        delta={'reference': 50, 'increasing': {'color': "#58a6ff"}, 'font': {'size': 9}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': chart_text_color, 'tickfont': {'color': chart_text_color, 'size': 8}},
            'bar': {'color': humid_color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': chart_text_color,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(210, 153, 34, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(63, 185, 80, 0.3)'},
                {'range': [60, 80], 'color': 'rgba(88, 166, 255, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(137, 87, 229, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#8957e5", 'width': 3},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig_humid.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': chart_text_color, 'family': 'Poppins'},
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        autosize=True
    )
    st.plotly_chart(fig_humid, use_container_width=True)

with col3:
    _, _, adc_raw, local_aqi, aqi_category = extract_sensor_values(st.session_state.sensor_data)
    # AQI gauge colors
    if local_aqi <= 50:
        bar_color = "#3fb950"
    elif local_aqi <= 100:
        bar_color = "#d29922"
    elif local_aqi <= 200:
        bar_color = "#f0883e"
    else:
        bar_color = "#f85149"
        
    fig_aqi = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=local_aqi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Air Quality Index", 'font': {'size': 12, 'color': chart_text_color, 'family': 'Poppins'}},
        number={'font': {'size': 22, 'color': chart_text_color, 'family': 'Poppins'}},
        delta={'reference': 50, 'increasing': {'color': "#f85149"}, 'font': {'size': 9}},
        gauge={
            'axis': {'range': [0, 500], 'tickcolor': chart_text_color, 'tickfont': {'color': chart_text_color, 'size': 8}},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': chart_text_color,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(63, 185, 80, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(210, 153, 34, 0.3)'},
                {'range': [100, 200], 'color': 'rgba(240, 136, 62, 0.3)'},
                {'range': [200, 300], 'color': 'rgba(248, 81, 73, 0.4)'},
                {'range': [300, 500], 'color': 'rgba(163, 21, 21, 0.5)'}
            ],
            'threshold': {
                'line': {'color': "#f85149", 'width': 3},
                'thickness': 0.75,
                'value': 150
            }
        }
    ))
    fig_aqi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': chart_text_color, 'family': 'Poppins'},
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        autosize=True
    )
    st.plotly_chart(fig_aqi, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Historical Charts
st.markdown(f'<p class="section-title">{SVG_ICONS["chart"]} Real-Time Data Trends</p>', unsafe_allow_html=True)

if len(st.session_state.data_history['time']) > 0:
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature', 'Humidity', 'Air Quality Index', 'All Sensors'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    times = list(st.session_state.data_history['time'])
    
    # Temperature
    fig.add_trace(
        go.Scatter(
            x=times,
            y=list(st.session_state.data_history['suhu']),
            mode='lines+markers',
            name='Temperature',
            line=dict(color=temp_color, width=2, shape='spline'),
            marker=dict(size=5, symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(240, 136, 62, 0.15)'
        ),
        row=1, col=1
    )
    
    # Humidity
    fig.add_trace(
        go.Scatter(
            x=times,
            y=list(st.session_state.data_history['lembab']),
            mode='lines+markers',
            name='Humidity',
            line=dict(color=humid_color, width=2, shape='spline'),
            marker=dict(size=5, symbol='diamond'),
            fill='tozeroy',
            fillcolor='rgba(88, 166, 255, 0.15)'
        ),
        row=1, col=2
    )
    
    # AQI
    fig.add_trace(
        go.Scatter(
            x=times,
            y=list(st.session_state.data_history['aqi']),
            mode='lines+markers',
            name='AQI',
            line=dict(color=aqi_color, width=2, shape='spline'),
            marker=dict(size=5, symbol='square'),
            fill='tozeroy',
            fillcolor='rgba(63, 185, 80, 0.15)'
        ),
        row=2, col=1
    )
    
    # All combined
    fig.add_trace(
        go.Scatter(x=times, y=list(st.session_state.data_history['suhu']),
                   mode='lines', name='Temp', line=dict(color=temp_color, width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=times, y=list(st.session_state.data_history['lembab']),
                   mode='lines', name='Humid', line=dict(color=humid_color, width=2)),
        row=2, col=2
    )
    
    # Theme-aware colors
    grid_color = 'rgba(255,255,255,0.06)' if st.session_state.theme == 'dark' else 'rgba(0,0,0,0.06)'
    
    fig.update_layout(
        height=650,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(22,27,34,0.5)' if st.session_state.theme == 'dark' else 'rgba(246,248,250,0.8)',
        font=dict(color=chart_text_color, family='Poppins'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=chart_text_color),
            bgcolor='rgba(0,0,0,0)'
        )
    )
    
    # Update axes with proper theme colors
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor=grid_color, 
        tickfont=dict(size=10, color=chart_text_color),
        linecolor=grid_color,
        zerolinecolor=grid_color
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor=grid_color, 
        tickfont=dict(size=10, color=chart_text_color),
        linecolor=grid_color,
        zerolinecolor=grid_color
    )
    
    # Update subplot titles color
    for annotation in fig.layout.annotations:
        annotation.font.color = chart_text_color
        annotation.font.family = 'Poppins'
        annotation.font.size = 13
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Waiting for sensor data... Please connect to MQTT broker first.")

# Data Table
st.markdown(f'<p class="section-title">{SVG_ICONS["table"]} Raw Data Log</p>', unsafe_allow_html=True)
if len(st.session_state.data_history['time']) > 0:
    import pandas as pd
    
    # Get current values using helper function
    _, _, adc_raw, local_aqi, aqi_category = extract_sensor_values(st.session_state.sensor_data)
    
    # Get AQI data
    aqi_data = {
        'adc_raw': st.session_state.sensor_data.get('adc_raw', 0),
        'adc_percent': st.session_state.sensor_data.get('adc_percent', 0),
        'local_aqi': st.session_state.sensor_data.get('local_aqi', 0),
        'aqi_category': st.session_state.sensor_data.get('aqi_category', 'N/A'),
        'baseline_adc': st.session_state.sensor_data.get('baseline_adc', 500),
        'weather_aqi': st.session_state.sensor_data.get('weather_aqi', 0)
    }
    
    # Create full dataframe for download
    df_full = pd.DataFrame({
        'Time': list(st.session_state.data_history['time']),
        'Temperature (°C)': list(st.session_state.data_history['suhu']),
        'Humidity (%)': list(st.session_state.data_history['lembab']),
        'AQI': list(st.session_state.data_history['aqi'])
    })
    
    # Display last 10 records
    df_display = pd.DataFrame({
        'Time': list(st.session_state.data_history['time'])[-10:],
        'Temp (°C)': [f"{x:.1f}" for x in list(st.session_state.data_history['suhu'])[-10:]],
        'Humid (%)': [f"{x:.1f}" for x in list(st.session_state.data_history['lembab'])[-10:]],
        'AQI': [f"{int(x)}" for x in list(st.session_state.data_history['aqi'])[-10:]],
        'ML Predict': [st.session_state.ml_prediction.get('status', 'N/A')] * min(10, len(list(st.session_state.data_history['time'])[-10:]))
    })
    df_display = df_display.iloc[::-1]  # Reverse to show newest first
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Download buttons
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = df_full.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        json_data = df_full.to_json(orient='records', date_format='iso')
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Show detailed AQI info in expander
    with st.expander("📊 Detail Air Quality Index", expanded=False):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.metric("Local AQI", f"{aqi_data.get('local_aqi', 0)}")
            st.metric("ADC Raw", f"{aqi_data.get('adc_raw', 0)}")
            st.metric("ADC Percent", f"{aqi_data.get('adc_percent', 0)}%")
        with col_a2:
            st.metric("Kategori", aqi_data.get('aqi_category', 'N/A'))
            st.metric("Baseline ADC", f"{aqi_data.get('baseline_adc', 500)}")
            weather_aqi = aqi_data.get('weather_aqi', 0)
            st.metric("AccuWeather AQI (Cilegon)", f"{weather_aqi}" if weather_aqi else "N/A")
        
        st.markdown("---")
        st.markdown("""
        **Skala AQI:**
        - 🟢 0-50: Baik
        - 🟡 51-100: Sedang
        - 🟠 101-150: Tidak Sehat (Sensitif)
        - 🔴 151-200: Tidak Sehat
        - 🟣 201-300: Sangat Tidak Sehat
        - ⚫ 301-500: Berbahaya
        """)


# Footer
st.markdown("---")
st.markdown(f"""
<div class="footer">
    <p style="font-size: 1.3rem; font-family: 'Poppins', sans-serif; font-weight: 700; margin-bottom: 15px">
        🌿 Smart Environment Monitor
    </p>
    <p style="font-size: 0.85rem; margin-top: 12px; color: {theme['text_secondary']};">
        Built with <a href="https://streamlit.io" target="_blank" style="color: {theme['accent']};">Streamlit</a> & 
        <a href="https://plotly.com" target="_blank" style="color: {theme['accent']};">Plotly</a>
    </p>
    <p style="font-size: 1rem; margin-top: 20px; font-weight: 600; color: {theme['text_primary']};">
        ✨ by Kita pergi hari ini
    </p>
</div>
""", unsafe_allow_html=True)
