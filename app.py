import os
import sys
import json
import uuid
import hashlib
import logging
import pickle
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import wraps

# Data Science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image

# Database
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor

# Streamlit
import streamlit as st
import streamlit.components.v1 as components

# Additional
import requests
from groq import Groq
import folium
from streamlit_folium import folium_static
import kaggle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration"""
    
    # Database configuration (PostgreSQL on Amazon Linux)
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'civic_complaints')
    DB_USER = os.getenv('DB_USER', 'civic_admin')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'secure_password_123')
    
    # API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
    KAGGLE_KEY = os.getenv('KAGGLE_KEY')
    
    # Application settings
    UPLOAD_DIR = Path('static/images')
    MODEL_DIR = Path('data/models')
    DATASET_DIR = Path('data/datasets')
    
    # Model settings
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # Complaint categories
    CATEGORIES = ['pothole', 'garbage', 'broken_streetlight', 'road_damage', 'other']
    CATEGORY_NAMES = {
        'pothole': 'Pothole',
        'garbage': 'Garbage Accumulation',
        'broken_streetlight': 'Broken Streetlight',
        'road_damage': 'Road Damage',
        'other': 'Other Issue'
    }
    
    # Status flow
    STATUSES = ['pending', 'verified', 'assigned', 'in_progress', 'resolved', 'rejected']
    STATUS_COLORS = {
        'pending': '#ffa500',
        'verified': '#3498db',
        'assigned': '#9b59b6',
        'in_progress': '#f39c12',
        'resolved': '#27ae60',
        'rejected': '#e74c3c'
    }

# Create directories
Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
Config.DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Database Connection Pool
# ============================================================================

class DatabasePool:
    """PostgreSQL connection pool manager"""
    
    _pool = None
    
    @classmethod
    def get_pool(cls):
        if cls._pool is None:
            try:
                cls._pool = pool.SimpleConnectionPool(
                    1, 20,
                    host=Config.DB_HOST,
                    port=Config.DB_PORT,
                    database=Config.DB_NAME,
                    user=Config.DB_USER,
                    password=Config.DB_PASSWORD
                )
                logger.info("Database connection pool created")
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                raise
        return cls._pool
    
    @classmethod
    def get_connection(cls):
        return cls.get_pool().getconn()
    
    @classmethod
    def return_connection(cls, conn):
        cls.get_pool().putconn(conn)

# ============================================================================
# Database Schema and Initialization
# ============================================================================

def init_database():
    """Initialize database schema"""
    conn = None
    try:
        conn = DatabasePool.get_connection()
        cur = conn.cursor()
        
        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                email VARCHAR(255),
                role VARCHAR(50) DEFAULT 'citizen',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        
        # Create complaints table
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
                assigned_to INTEGER REFERENCES users(id),
                resolution_notes TEXT
            )
        """)
        
        # Create status_history table
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
        
        # Create analytics table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100),
                metric_value JSONB,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_complaints_status ON complaints(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_complaints_category ON complaints(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_complaints_location ON complaints(latitude, longitude)")
        
        conn.commit()
        logger.info("Database schema initialized")
        
        # Create default admin if not exists
        create_default_admin()
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            DatabasePool.return_connection(conn)

def create_default_admin():
    """Create default admin user"""
    conn = None
    try:
        conn = DatabasePool.get_connection()
        cur = conn.cursor()
        
        # Check if admin exists
        cur.execute("SELECT id FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            password_hash = hashlib.sha256("admin123".encode()).hexdigest()
            cur.execute("""
                INSERT INTO users (username, password_hash, email, role)
                VALUES (%s, %s, %s, %s)
            """, ('admin', password_hash, 'admin@civic.com', 'admin'))
            conn.commit()
            logger.info("Default admin user created")
            
    except Exception as e:
        logger.error(f"Error creating default admin: {e}")
    finally:
        if conn:
            DatabasePool.return_connection(conn)

# ============================================================================
# Authentication & Authorization
# ============================================================================

class AuthManager:
    """Authentication and session management"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[Dict]:
        conn = None
        try:
            conn = DatabasePool.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            password_hash = AuthManager.hash_password(password)
            cur.execute("""
                SELECT id, username, email, role FROM users
                WHERE username = %s AND password_hash = %s
            """, (username, password_hash))
            
            user = cur.fetchone()
            
            if user:
                # Update last login
                cur.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (user['id'],))
                conn.commit()
                
            return user
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
        finally:
            if conn:
                DatabasePool.return_connection(conn)
    
    @staticmethod
    def register(username: str, password: str, email: str) -> bool:
        conn = None
        try:
            conn = DatabasePool.get_connection()
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
                DatabasePool.return_connection(conn)
    
    @staticmethod
    def require_auth(role: Optional[str] = None):
        """Decorator for role-based access control"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if 'user' not in st.session_state:
                    st.error("Please login to access this page")
                    st.stop()
                
                if role and st.session_state.user.get('role') != role:
                    st.error(f"Access denied. {role.capitalize()} role required")
                    st.stop()
                    
                return func(*args, **kwargs)
            return wrapper
        return decorator

# ============================================================================
# CNN Model for Image Classification
# ============================================================================

class ComplaintClassifier:
    """CNN-based complaint image classifier"""
    
    def __init__(self):
        self.model = None
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        model_path = Config.MODEL_DIR / 'complaint_classifier.h5'
        
        if model_path.exists():
            try:
                self.model = tf.keras.models.load_model(model_path)
                logger.info("Model loaded from disk")
                return
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        
        # Create new model
        self.create_model()
    
    def create_model(self):
        """Create CNN model architecture"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*Config.IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(Config.CATEGORIES), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        logger.info("New model created")
    
    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """Preprocess image for prediction"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, Config.IMG_SIZE)
        img = img / 255.0  # Normalize
        return np.expand_dims(img, axis=0)
    
    def predict(self, image_path: Path) -> Tuple[str, float, Dict]:
        """Predict complaint category with confidence scores"""
        processed_img = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_img)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        category = Config.CATEGORIES[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Get probability distribution
        probabilities = {
            cat: float(predictions[i]) 
            for i, cat in enumerate(Config.CATEGORIES)
        }
        
        return category, confidence, probabilities
    
    def train_on_kaggle_dataset(self):
        """Train model using Kaggle dataset"""
        # Download dataset from Kaggle
        try:
            kaggle.api.dataset_download_files(
                'anandhuh/pothole-image-dataset',
                path=str(Config.DATASET_DIR),
                unzip=True
            )
            logger.info("Kaggle dataset downloaded")
        except Exception as e:
            logger.warning(f"Could not download Kaggle dataset: {e}")
            # Use synthetic data for demonstration
            self.train_with_synthetic_data()
            return
        
        # Load and prepare dataset
        # (Implementation depends on actual dataset structure)
        pass
    
    def train_with_synthetic_data(self):
        """Generate synthetic training data for demonstration"""
        # This is a placeholder for actual training
        # In production, you would use real datasets
        logger.info("Using synthetic data for training")
        
        # Create dummy data for demonstration
        X = np.random.rand(1000, *Config.IMG_SIZE, 3)
        y = np.random.randint(0, len(Config.CATEGORIES), 1000)
        y = tf.keras.utils.to_categorical(y, len(Config.CATEGORIES))
        
        # Train for a few epochs
        self.model.fit(
            X, y,
            epochs=2,
            batch_size=Config.BATCH_SIZE,
            validation_split=0.2,
            verbose=1
        )
        
        # Save model
        self.model.save(Config.MODEL_DIR / 'complaint_classifier.h5')

# Initialize classifier
classifier = ComplaintClassifier()

# ============================================================================
# Groq LLM Integration with RAG
# ============================================================================

class GroqLLM:
    """LLM integration using Groq API"""
    
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
    
    def generate_summary(self, complaint: Dict) -> str:
        """Generate summary for complaint"""
        prompt = f"""
        Generate a concise summary for this civic complaint:
        Category: {complaint['category']}
        Description: {complaint.get('description', 'No description provided')}
        Location: {complaint.get('address', 'Unknown')}
        
        Summary should be 1-2 sentences highlighting the key issue.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM summary error: {e}")
            return f"{complaint['category']} issue reported at {complaint.get('address', 'unknown location')}"
    
    def suggest_resolution(self, complaint: Dict) -> str:
        """Suggest resolution based on complaint type"""
        prompt = f"""
        Suggest a resolution for this civic complaint:
        Category: {complaint['category']}
        Description: {complaint.get('description', 'No description provided')}
        
        Provide 2-3 actionable steps for municipal authorities to resolve this issue.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.5,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM resolution error: {e}")
            return "Standard procedure: Verify complaint, assign to appropriate department, and schedule resolution."
    
    def rag_query(self, query: str) -> str:
        """Retrieval Augmented Generation - query past complaints"""
        conn = None
        try:
            # Retrieve relevant past complaints
            conn = DatabasePool.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT category, description, resolution_notes, status
                FROM complaints
                WHERE status = 'resolved'
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            past_complaints = cur.fetchall()
            
            # Build context
            context = "\n".join([
                f"- {c['category']}: {c['description'][:100]}... Resolved: {c.get('resolution_notes', 'N/A')}"
                for c in past_complaints
            ])
            
            # Query LLM with context
            prompt = f"""
            Based on past resolved complaints:
            {context}
            
            Answer this query: {query}
            
            Provide insights on how similar issues were handled.
            """
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return "Unable to retrieve past complaint data."
        finally:
            if conn:
                DatabasePool.return_connection(conn)

# Initialize Groq client
groq_client = GroqLLM() if Config.GROQ_API_KEY else None

# ============================================================================
# Complaint Management
# ============================================================================

class ComplaintManager:
    """CRUD operations for complaints"""
    
    @staticmethod
    def create_complaint(data: Dict) -> str:
        """Create new complaint"""
        conn = None
        try:
            conn = DatabasePool.get_connection()
            cur = conn.cursor()
            
            # Generate unique complaint ID
            complaint_id = f"CMP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
            
            cur.execute("""
                INSERT INTO complaints (
                    complaint_id, user_id, category, description, 
                    image_path, latitude, longitude, address, 
                    confidence_score, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
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
            logger.info(f"Complaint created: {complaint_id}")
            return complaint_id
            
        except Exception as e:
            logger.error(f"Error creating complaint: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                DatabasePool.return_connection(conn)
    
    @staticmethod
    def get_complaint(complaint_id: str) -> Optional[Dict]:
        """Get complaint by ID"""
        conn = None
        try:
            conn = DatabasePool.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT c.*, u.username as reporter_name
                FROM complaints c
                LEFT JOIN users u ON c.user_id = u.id
                WHERE c.complaint_id = %s
            """, (complaint_id,))
            
            return cur.fetchone()
            
        except Exception as e:
            logger.error(f"Error fetching complaint: {e}")
            return None
        finally:
            if conn:
                DatabasePool.return_connection(conn)
    
    @staticmethod
    def update_status(complaint_id: str, new_status: str, user_id: int, notes: str = None):
        """Update complaint status with history tracking"""
        conn = None
        try:
            conn = DatabasePool.get_connection()
            cur = conn.cursor()
            
            # Get current status
            cur.execute("SELECT status FROM complaints WHERE complaint_id = %s", (complaint_id,))
            result = cur.fetchone()
            if not result:
                raise ValueError("Complaint not found")
            
            old_status = result[0]
            
            # Update complaint
            cur.execute("""
                UPDATE complaints 
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE complaint_id = %s
            """, (new_status, complaint_id))
            
            # Record status change
            cur.execute("""
                INSERT INTO status_history (complaint_id, old_status, new_status, changed_by, notes)
                VALUES (%s, %s, %s, %s, %s)
            """, (complaint_id, old_status, new_status, user_id, notes))
            
            conn.commit()
            logger.info(f"Status updated: {complaint_id} -> {new_status}")
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                DatabasePool.return_connection(conn)
    
    @staticmethod
    def get_complaints(filters: Dict = None) -> List[Dict]:
        """Get complaints with filters"""
        conn = None
        try:
            conn = DatabasePool.get_connection()
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
                if filters.get('category'):
                    query += " AND c.category = %s"
                    params.append(filters['category'])
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
                DatabasePool.return_connection(conn)

# ============================================================================
# Analytics & Statistics
# ============================================================================

class AnalyticsEngine:
    """Statistical analysis and visualization"""
    
    @staticmethod
    def compute_descriptive_stats(complaints: List[Dict]) -> Dict:
        """Compute descriptive statistics"""
        if not complaints:
            return {}
        
        df = pd.DataFrame(complaints)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        stats_dict = {
            'total_complaints': len(df),
            'avg_response_time': None,
            'resolution_rate': (df['status'] == 'resolved').sum() / len(df) * 100,
            'categories_distribution': df['category'].value_counts().to_dict(),
            'status_distribution': df['status'].value_counts().to_dict(),
            'daily_rate': df.groupby(df['created_at'].dt.date).size().to_dict()
        }
        
        return stats_dict
    
    @staticmethod
    def perform_clustering(complaints: List[Dict]) -> pd.DataFrame:
        """Cluster complaints by location using K-means"""
        if not complaints:
            return pd.DataFrame()
        
        # Extract location data
        locations = []
        for c in complaints:
            if c.get('latitude') and c.get('longitude'):
                locations.append([c['latitude'], c['longitude']])
        
        if len(locations) < 10:
            return pd.DataFrame()
        
        df_locations = pd.DataFrame(locations, columns=['latitude', 'longitude'])
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_locations)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=min(5, len(locations) // 3), random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        df_locations['cluster'] = clusters
        return df_locations
    
    @staticmethod
    def hypothesis_testing(complaints: List[Dict]) -> Dict:
        """Perform statistical hypothesis tests"""
        if not complaints:
            return {}
        
        df = pd.DataFrame(complaints)
        
        # Test if potholes take longer to resolve than other complaints
        df['resolution_time'] = pd.to_datetime(df['updated_at']) - pd.to_datetime(df['created_at'])
        df['resolution_hours'] = df['resolution_time'].dt.total_seconds() / 3600
        
        pothole_times = df[df['category'] == 'pothole']['resolution_hours'].dropna()
        other_times = df[df['category'] != 'pothole']['resolution_hours'].dropna()
        
        if len(pothole_times) > 1 and len(other_times) > 1:
            t_stat, p_value = stats.ttest_ind(pothole_times, other_times, nan_policy='omit')
            
            return {
                'test_name': 'T-test: Pothole resolution time vs Others',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_pothole': pothole_times.mean(),
                'mean_others': other_times.mean()
            }
        
        return {}
    
    @staticmethod
    def create_visualizations(complaints: List[Dict]):
        """Create data visualizations"""
        if not complaints:
            return None, None
        
        df = pd.DataFrame(complaints)
        
        # Category distribution
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        df['category'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title('Complaints by Category')
        ax1.set_xlabel('Category')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Status distribution
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        df['status'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        ax2.set_title('Complaints by Status')
        
        return fig1, fig2
    
    @staticmethod
    def bayesian_priority_score(complaint: Dict) -> float:
        """Calculate Bayesian priority score for complaint"""
        # Prior probabilities based on category
        category_prior = {
            'pothole': 0.3,
            'garbage': 0.2,
            'broken_streetlight': 0.25,
            'road_damage': 0.2,
            'other': 0.05
        }
        
        # Likelihood based on confidence score
        confidence = complaint.get('confidence_score', 0.5)
        likelihood = confidence
        
        # Evidence (age of complaint)
        created_at = complaint.get('created_at')
        if created_at:
            age_days = (datetime.now() - created_at).days
            evidence = 1 / (1 + np.exp(-age_days / 7))  # Logistic function
        else:
            evidence = 0.5
        
        # Calculate posterior probability
        prior = category_prior.get(complaint.get('category', 'other'), 0.1)
        posterior = (likelihood * prior) / evidence
        
        return min(posterior, 1.0)

# ============================================================================
# Streamlit UI Components
# ============================================================================

def setup_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title="Smart Civic Complaint System",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
            padding: 0.25rem 0.5rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: bold;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

def login_page():
    """User login interface"""
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
    """Citizen dashboard"""
    st.markdown('<div class="main-header"><h1>🏙️ Citizen Dashboard</h1><p>Report and track complaints</p></div>', unsafe_allow_html=True)
    
    # Get user's complaints
    user_complaints = ComplaintManager.get_complaints({'user_id': st.session_state.user['id']})
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.expander("📝 Report New Issue", expanded=True):
            with st.form("report_form"):
                # Category selection with probability
                st.write("**Issue Category**")
                categories = Config.CATEGORY_NAMES
                selected_category = st.selectbox("Select category", list(categories.keys()))
                
                # Description
                description = st.text_area("Description", placeholder="Describe the issue in detail...")
                
                # Image upload
                uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
                
                # Location (auto-detect or manual)
                location_method = st.radio("Location", ["Auto-detect", "Manual"])
                
                latitude = None
                longitude = None
                address = None
                
                if location_method == "Auto-detect":
                    st.info("Location will be captured from your browser")
                    # In production, use browser geolocation API
                    # For demo, use default coordinates
                    latitude = 12.9716
                    longitude = 77.5946
                    address = "Bengaluru, India"
                else:
                    latitude = st.number_input("Latitude", value=12.9716, format="%.6f")
                    longitude = st.number_input("Longitude", value=77.5946, format="%.6f")
                    address = st.text_input("Address", placeholder="Enter address")
                
                submitted = st.form_submit_button("Submit Complaint")
                
                if submitted and uploaded_file:
                    # Save image
                    image_path = Config.UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Classify image
                    category, confidence, probabilities = classifier.predict(image_path)
                    
                    # Show prediction
                    st.info(f"AI Analysis: **{Config.CATEGORY_NAMES[category]}** detected with {confidence:.1%} confidence")
                    
                    # Show probability distribution
                    prob_df = pd.DataFrame(list(probabilities.items()), columns=['Category', 'Probability'])
                    prob_df['Category'] = prob_df['Category'].map(Config.CATEGORY_NAMES)
                    st.bar_chart(prob_df.set_index('Category'))
                    
                    # Create complaint
                    complaint_data = {
                        'user_id': st.session_state.user['id'],
                        'category': category,
                        'description': description,
                        'image_path': image_path,
                        'latitude': latitude,
                        'longitude': longitude,
                        'address': address,
                        'confidence_score': confidence
                    }
                    
                    complaint_id = ComplaintManager.create_complaint(complaint_data)
                    
                    # Generate summary using LLM
                    if groq_client:
                        summary = groq_client.generate_summary(complaint_data)
                        st.success(f"Complaint submitted! ID: {complaint_id}")
                        st.info(f"AI Summary: {summary}")
                    else:
                        st.success(f"Complaint submitted! ID: {complaint_id}")
                    
                    st.rerun()
    
    with col2:
        st.subheader("📊 My Complaints")
        
        if not user_complaints:
            st.info("No complaints reported yet")
        else:
            for complaint in user_complaints:
                status_color = Config.STATUS_COLORS.get(complaint['status'], '#95a5a6')
                
                with st.container():
                    st.markdown(f"""
                        <div class="complaint-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong>#{complaint['complaint_id']}</strong>
                                <span class="status-badge" style="background: {status_color}; color: white;">
                                    {complaint['status'].upper()}
                                </span>
                            </div>
                            <p><strong>Category:</strong> {Config.CATEGORY_NAMES.get(complaint['category'], complaint['category'])}</p>
                            <p><strong>Reported:</strong> {complaint['created_at'].strftime('%Y-%m-%d %H:%M')}</p>
                            <p><strong>Location:</strong> {complaint.get('address', 'Unknown')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Status tracking timeline
                if complaint['status'] != 'pending':
                    st.caption(f"Last updated: {complaint['updated_at'].strftime('%Y-%m-%d %H:%M')}")

def admin_dashboard():
    """Admin dashboard"""
    st.markdown('<div class="main-header"><h1>👨‍💼 Admin Dashboard</h1><p>Manage and analyze complaints</p></div>', unsafe_allow_html=True)
    
    # Get all complaints
    complaints = ComplaintManager.get_complaints()
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Complaints Management", "Analytics", "AI Insights"])
    
    with tab1:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(complaints)
        resolved = sum(1 for c in complaints if c['status'] == 'resolved')
        pending = sum(1 for c in complaints if c['status'] == 'pending')
        
        with col1:
            st.metric("Total Complaints", total)
        with col2:
            st.metric("Resolved", resolved, delta=f"{resolved/total*100:.1f}%" if total > 0 else "0%")
        with col3:
            st.metric("Pending", pending)
        with col4:
            resolution_rate = (resolved / total * 100) if total > 0 else 0
            st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        
        # Charts
        fig1, fig2 = AnalyticsEngine.create_visualizations(complaints)
        if fig1 and fig2:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig1)
            with col2:
                st.pyplot(fig2)
        
        # Location clustering
        st.subheader("Complaint Hotspots")
        clusters = AnalyticsEngine.perform_clustering(complaints)
        if not clusters.empty:
            st.write("Complaints have been clustered into zones for better resource allocation:")
            st.dataframe(clusters.value_counts().reset_index().rename(columns={0: 'count'}))
    
    with tab2:
        st.subheader("Manage Complaints")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All"] + Config.STATUSES)
        with col2:
            category_filter = st.selectbox("Filter by Category", ["All"] + list(Config.CATEGORY_NAMES.values()))
        
        # Display complaints
        filtered_complaints = complaints
        if status_filter != "All":
            filtered_complaints = [c for c in filtered_complaints if c['status'] == status_filter]
        if category_filter != "All":
            # Map display name back to key
            cat_key = {v: k for k, v in Config.CATEGORY_NAMES.items()}[category_filter]
            filtered_complaints = [c for c in filtered_complaints if c['category'] == cat_key]
        
        for complaint in filtered_complaints:
            with st.expander(f"#{complaint['complaint_id']} - {Config.CATEGORY_NAMES.get(complaint['category'], complaint['category'])}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {complaint.get('description', 'No description')}")
                    st.write(f"**Reported by:** {complaint.get('reporter_name', 'Unknown')}")
                    st.write(f"**Location:** {complaint.get('address', 'Not specified')}")
                    st.write(f"**Date:** {complaint['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    
                    if complaint.get('confidence_score'):
                        st.write(f"**AI Confidence:** {complaint['confidence_score']:.1%}")
                
                with col2:
                    # Status update
                    new_status = st.selectbox(
                        "Update Status",
                        Config.STATUSES,
                        index=Config.STATUSES.index(complaint['status']),
                        key=f"status_{complaint['complaint_id']}"
                    )
                    
                    notes = st.text_area("Notes", key=f"notes_{complaint['complaint_id']}")
                    
                    if st.button("Update", key=f"update_{complaint['complaint_id']}"):
                        ComplaintManager.update_status(
                            complaint['complaint_id'],
                            new_status,
                            st.session_state.user['id'],
                            notes
                        )
                        st.success("Status updated!")
                        st.rerun()
                    
                    # Show resolution suggestion
                    if groq_client and complaint['status'] == 'pending':
                        if st.button("Get AI Suggestion", key=f"suggest_{complaint['complaint_id']}"):
                            suggestion = groq_client.suggest_resolution(complaint)
                            st.info(f"**AI Suggestion:** {suggestion}")
    
    with tab3:
        st.subheader("Advanced Analytics")
        
        # Descriptive statistics
        stats = AnalyticsEngine.compute_descriptive_stats(complaints)
        if stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Descriptive Statistics**")
                st.metric("Total Complaints", stats['total_complaints'])
                st.metric("Resolution Rate", f"{stats['resolution_rate']:.1f}%")
                st.write("**Category Distribution:**")
                for cat, count in stats['categories_distribution'].items():
                    st.write(f"- {Config.CATEGORY_NAMES.get(cat, cat)}: {count}")
            
            with col2:
                # Hypothesis testing
                test_results = AnalyticsEngine.hypothesis_testing(complaints)
                if test_results:
                    st.write("**Hypothesis Testing Results**")
                    st.write(f"Test: {test_results['test_name']}")
                    st.write(f"P-value: {test_results['p_value']:.4f}")
                    st.write(f"Significant: {'Yes' if test_results['significant'] else 'No'}")
                    if 'mean_pothole' in test_results:
                        st.write(f"Mean resolution time (potholes): {test_results['mean_pothole']:.1f} hours")
                        st.write(f"Mean resolution time (others): {test_results['mean_others']:.1f} hours")
    
    with tab4:
        st.subheader("AI-Powered Insights")
        
        if groq_client:
            # RAG query interface
            st.write("**Query Past Complaints**")
            query = st.text_input("Ask about past complaints (e.g., 'How are potholes typically resolved?')")
            if query:
                with st.spinner("Retrieving information..."):
                    response = groq_client.rag_query(query)
                    st.info(response)
            
            # Bayesian priority analysis
            st.write("**Bayesian Priority Analysis**")
            for complaint in complaints[:10]:  # Show top 10
                priority = AnalyticsEngine.bayesian_priority_score(complaint)
                if priority > 0.6:  # High priority
                    st.warning(f"🔴 #{complaint['complaint_id']}: High priority score {priority:.2f}")
                elif priority > 0.3:
                    st.info(f"🟡 #{complaint['complaint_id']}: Medium priority score {priority:.2f}")
        else:
            st.warning("Groq API key not configured. AI features are disabled.")

def main():
    """Main application entry point"""
    setup_page()
    
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Initialize database
    init_database()
    
    # Render appropriate view
    if not st.session_state.user:
        login_page()
    else:
        # Sidebar
        with st.sidebar:
            st.image("https://img.icons8.com/fluency/96/000000/complaint.png", width=50)
            st.write(f"Welcome, **{st.session_state.user['username']}**")
            st.write(f"Role: {st.session_state.user['role'].capitalize()}")
            st.divider()
            
            if st.session_state.user['role'] == 'admin':
                st.page_link("app.py", label="🏠 Dashboard", icon="📊")
            else:
                st.page_link("app.py", label="🏠 Dashboard", icon="📝")
            
            if st.button("🚪 Logout"):
                st.session_state.user = None
                st.rerun()
        
        # Main content
        if st.session_state.user['role'] == 'admin':
            admin_dashboard()
        else:
            citizen_dashboard()

if __name__ == "__main__":
    main()
