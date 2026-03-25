"""
Smart Civic Complaint System - Lightweight Version
No TensorFlow required - uses simple image analysis
"""

import os
import uuid
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Data Science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Image processing
try:
    from PIL import Image
    import cv2
except ImportError:
    Image = None
    cv2 = None

# Database
import psycopg2
from psycopg2.extras import RealDictCursor

# Streamlit
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'civic_complaints')
    DB_USER = os.getenv('DB_USER', 'civic_admin')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'secure_password_123')
    
    UPLOAD_DIR = Path('static/images')
    
    CATEGORIES = ['pothole', 'garbage', 'broken_streetlight', 'road_damage', 'other']
    CATEGORY_NAMES = {
        'pothole': 'Pothole',
        'garbage': 'Garbage Accumulation',
        'broken_streetlight': 'Broken Streetlight',
        'road_damage': 'Road Damage',
        'other': 'Other Issue'
    }
    
    STATUSES = ['pending', 'verified', 'assigned', 'in_progress', 'resolved', 'rejected']
    STATUS_COLORS = {
        'pending': '#ffa500',
        'verified': '#3498db',
        'assigned': '#9b59b6',
        'in_progress': '#f39c12',
        'resolved': '#27ae60',
        'rejected': '#e74c3c'
    }

Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Database Connection
# ============================================================================

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

# ============================================================================
# Database Schema
# ============================================================================

def init_database():
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                email VARCHAR(255),
                role VARCHAR(50) DEFAULT 'citizen',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id SERIAL PRIMARY KEY,
                complaint_id VARCHAR(50) UNIQUE NOT NULL,
                user_id INTEGER REFERENCES users(id),
                category VARCHAR(50) NOT NULL,
                description TEXT,
                image_path VARCHAR(500),
                latitude FLOAT,
                longitude FLOAT,
                address TEXT,
                status VARCHAR(50) DEFAULT 'pending',
                confidence_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolution_notes TEXT
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS status_history (
                id SERIAL PRIMARY KEY,
                complaint_id VARCHAR(50) REFERENCES complaints(complaint_id),
                old_status VARCHAR(50),
                new_status VARCHAR(50),
                changed_by INTEGER REFERENCES users(id),
                notes TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        
        # Create default admin
        cur.execute("SELECT id FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            password_hash = hashlib.sha256("admin123".encode()).hexdigest()
            cur.execute("""
                INSERT INTO users (username, password_hash, email, role)
                VALUES (%s, %s, %s, %s)
            """, ('admin', password_hash, 'admin@civic.com', 'admin'))
            conn.commit()
        
        return True
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False
    finally:
        if conn:
            conn.close()

# ============================================================================
# Authentication
# ============================================================================

class AuthManager:
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[Dict]:
        conn = get_db_connection()
        if not conn:
            return None
        
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            password_hash = AuthManager.hash_password(password)
            cur.execute("""
                SELECT id, username, email, role FROM users
                WHERE username = %s AND password_hash = %s
            """, (username, password_hash))
            return cur.fetchone()
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def register(username: str, password: str, email: str) -> bool:
        conn = get_db_connection()
        if not conn:
            return False
        
        try:
            cur = conn.cursor()
            password_hash = AuthManager.hash_password(password)
            cur.execute("""
                INSERT INTO users (username, password_hash, email, role)
                VALUES (%s, %s, %s, %s)
            """, (username, password_hash, email, 'citizen'))
            conn.commit()
            return True
        except psycopg2.IntegrityError:
            return False
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
        finally:
            if conn:
                conn.close()

# ============================================================================
# Image Analysis (Simple, no TensorFlow)
# ============================================================================

def analyze_image(image_path):
    """Simple image analysis without TensorFlow"""
    try:
        # Try to load with PIL first
        if Image:
            img = Image.open(image_path)
            img = img.resize((100, 100))
            img_array = np.array(img)
            
            # Calculate basic features
            brightness = np.mean(img_array) / 255.0
            contrast = np.std(img_array) / 255.0
            
            # Simple rule-based classification
            if brightness < 0.3:
                return 'pothole', 0.75
            elif brightness > 0.7:
                return 'garbage', 0.70
            elif contrast > 0.2:
                return 'road_damage', 0.65
            else:
                return 'other', 0.55
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
    
    return 'other', 0.5

# ============================================================================
# Complaint Management
# ============================================================================

class ComplaintManager:
    @staticmethod
    def create_complaint(data: Dict) -> str:
        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")
        
        try:
            cur = conn.cursor()
            complaint_id = f"CMP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
            
            cur.execute("""
                INSERT INTO complaints (
                    complaint_id, user_id, category, description, 
                    image_path, latitude, longitude, address, 
                    confidence_score, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                complaint_id,
                data['user_id'],
                data['category'],
                data.get('description', ''),
                str(data.get('image_path', '')),
                data.get('latitude'),
                data.get('longitude'),
                data.get('address', ''),
                data.get('confidence_score'),
                'pending'
            ))
            
            conn.commit()
            return complaint_id
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def get_complaints(filters: Dict = None) -> List[Dict]:
        conn = get_db_connection()
        if not conn:
            return []
        
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            query = """
                SELECT c.*, u.username as reporter_name
                FROM complaints c
                LEFT JOIN users u ON c.user_id = u.id
                WHERE 1=1
            """
            params = []
            
            if filters:
                if filters.get('status'):
                    query += " AND c.status = %s"
                    params.append(filters['status'])
                if filters.get('user_id'):
                    query += " AND c.user_id = %s"
                    params.append(filters['user_id'])
            
            query += " ORDER BY c.created_at DESC"
            
            cur.execute(query, params)
            return cur.fetchall()
        except Exception as e:
            logger.error(f"Error fetching complaints: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def update_status(complaint_id: str, new_status: str, user_id: int, notes: str = None):
        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")
        
        try:
            cur = conn.cursor()
            cur.execute("SELECT status FROM complaints WHERE complaint_id = %s", (complaint_id,))
            result = cur.fetchone()
            if not result:
                raise ValueError("Complaint not found")
            
            old_status = result[0]
            
            cur.execute("""
                UPDATE complaints 
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE complaint_id = %s
            """, (new_status, complaint_id))
            
            cur.execute("""
                INSERT INTO status_history (complaint_id, old_status, new_status, changed_by, notes)
                VALUES (%s, %s, %s, %s, %s)
            """, (complaint_id, old_status, new_status, user_id, notes))
            
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

# ============================================================================
# Analytics
# ============================================================================

def compute_stats(complaints: List[Dict]) -> Dict:
    if not complaints:
        return {}
    
    df = pd.DataFrame(complaints)
    return {
        'total': len(df),
        'resolved': (df['status'] == 'resolved').sum(),
        'pending': (df['status'] == 'pending').sum(),
        'in_progress': (df['status'] == 'in_progress').sum(),
        'categories': df['category'].value_counts().to_dict(),
        'resolution_rate': (df['status'] == 'resolved').sum() / len(df) * 100 if len(df) > 0 else 0
    }

def create_charts(complaints: List[Dict]):
    if not complaints:
        return None, None
    
    df = pd.DataFrame(complaints)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    df['category'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title('Complaints by Category')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    df['status'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_title('Complaints by Status')
    plt.tight_layout()
    
    return fig1, fig2

# ============================================================================
# Streamlit UI
# ============================================================================

def setup_page():
    st.set_page_config(
        page_title="Smart Civic Complaint System",
        page_icon="🏙️",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .complaint-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

def login_page():
    st.markdown('<div class="main-header"><h1>🏙️ Smart Civic Complaint System</h1><p>Report urban issues instantly</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    user = AuthManager.authenticate(username, password)
                    if user:
                        st.session_state.user = user
                        st.success(f"Welcome {user['username']}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username")
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Register")
                
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif AuthManager.register(new_username, new_password, new_email):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")

def citizen_dashboard():
    st.markdown('<div class="main-header"><h1>🏙️ Citizen Dashboard</h1><p>Report and track complaints</p></div>', unsafe_allow_html=True)
    
    user_complaints = ComplaintManager.get_complaints({'user_id': st.session_state.user['id']})
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.expander("📝 Report New Issue", expanded=True):
            with st.form("report_form"):
                description = st.text_area("Description", placeholder="Describe the issue in detail...")
                uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
                address = st.text_input("Address/Location", placeholder="Enter address or location description")
                submitted = st.form_submit_button("Submit Complaint")
                
                if submitted and uploaded_file and address:
                    image_path = Config.UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner("Analyzing image..."):
                        try:
                            category, confidence = analyze_image(image_path)
                            st.success(f"AI Analysis: **{Config.CATEGORY_NAMES[category]}** detected with {confidence:.1%} confidence")
                        except Exception as e:
                            st.warning(f"Analysis failed: {e}")
                            category = 'other'
                            confidence = 0.5
                    
                    complaint_data = {
                        'user_id': st.session_state.user['id'],
                        'category': category,
                        'description': description,
                        'image_path': image_path,
                        'latitude': None,
                        'longitude': None,
                        'address': address,
                        'confidence_score': confidence
                    }
                    
                    try:
                        complaint_id = ComplaintManager.create_complaint(complaint_data)
                        st.success(f"✅ Complaint submitted! Tracking ID: {complaint_id}")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                elif submitted:
                    st.warning("Please fill all required fields")
    
    with col2:
        st.subheader("📊 My Complaints")
        
        if not user_complaints:
            st.info("No complaints reported yet")
        else:
            for complaint in user_complaints:
                status_color = Config.STATUS_COLORS.get(complaint['status'], '#95a5a6')
                st.markdown(f"""
                    <div class="complaint-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>#{complaint['complaint_id']}</strong>
                            <span class="status-badge" style="background: {status_color}; color: white;">
                                {complaint['status'].upper()}
                            </span>
                        </div>
                        <p><strong>Category:</strong> {Config.CATEGORY_NAMES.get(complaint['category'], complaint['category'])}</p>
                        <p><strong>Description:</strong> {complaint.get('description', 'No description')[:100]}</p>
                        <p><strong>Reported:</strong> {complaint['created_at'].strftime('%Y-%m-%d %H:%M') if complaint['created_at'] else 'N/A'}</p>
                        <p><strong>Location:</strong> {complaint.get('address', 'Unknown')}</p>
                    </div>
                """, unsafe_allow_html=True)

def admin_dashboard():
    st.markdown('<div class="main-header"><h1>👨‍💼 Admin Dashboard</h1><p>Manage and analyze complaints</p></div>', unsafe_allow_html=True)
    
    complaints = ComplaintManager.get_complaints()
    
    tab1, tab2, tab3 = st.tabs(["Overview", "Complaints Management", "Analytics"])
    
    with tab1:
        stats = compute_stats(complaints)
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Complaints", stats['total'])
            with col2:
                st.metric("Resolved", stats['resolved'])
            with col3:
                st.metric("Pending", stats.get('pending', 0))
            with col4:
                st.metric("Resolution Rate", f"{stats['resolution_rate']:.1f}%")
            
            fig1, fig2 = create_charts(complaints)
            if fig1 and fig2:
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig1)
                with col2:
                    st.pyplot(fig2)
    
    with tab2:
        st.subheader("Manage Complaints")
        
        status_filter = st.selectbox("Filter by Status", ["All"] + Config.STATUSES)
        
        filtered = complaints
        if status_filter != "All":
            filtered = [c for c in complaints if c['status'] == status_filter]
        
        for complaint in filtered:
            with st.expander(f"#{complaint['complaint_id']} - {Config.CATEGORY_NAMES.get(complaint['category'], complaint['category'])}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {complaint.get('description', 'No description')}")
                    st.write(f"**Reported by:** {complaint.get('reporter_name', 'Unknown')}")
                    st.write(f"**Location:** {complaint.get('address', 'Not specified')}")
                    if complaint.get('created_at'):
                        st.write(f"**Date:** {complaint['created_at'].strftime('%Y-%m-%d %H:%M')}")
                
                with col2:
                    new_status = st.selectbox(
                        "Update Status",
                        Config.STATUSES,
                        index=Config.STATUSES.index(complaint['status']),
                        key=f"status_{complaint['complaint_id']}"
                    )
                    notes = st.text_area("Resolution Notes", key=f"notes_{complaint['complaint_id']}")
                    
                    if st.button("Update", key=f"update_{complaint['complaint_id']}"):
                        try:
                            ComplaintManager.update_status(
                                complaint['complaint_id'],
                                new_status,
                                st.session_state.user['id'],
                                notes
                            )
                            st.success("Status updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    with tab3:
        st.subheader("Statistics")
        stats = compute_stats(complaints)
        if stats:
            st.write("**Category Distribution:**")
            for cat, count in stats.get('categories', {}).items():
                st.write(f"- {Config.CATEGORY_NAMES.get(cat, cat)}: {count}")

def main():
    setup_page()
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if not init_database():
        st.error("Database connection failed! Please ensure PostgreSQL is running.")
        return
    
    if not st.session_state.user:
        login_page()
    else:
        with st.sidebar:
            st.write(f"Welcome, **{st.session_state.user['username']}**")
            st.write(f"Role: {st.session_state.user['role'].capitalize()}")
            st.divider()
            if st.button("🚪 Logout"):
                st.session_state.user = None
                st.rerun()
        
        if st.session_state.user['role'] == 'admin':
            admin_dashboard()
        else:
            citizen_dashboard()

if __name__ == "__main__":
    main()
