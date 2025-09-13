import torch
import cv2
import numpy as np
from PIL import Image
import json
import sqlite3
from typing import List, Dict, Tuple, Optional
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from datetime import datetime

class SignatureExtractor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize the signature extractor with specified Qwen model
        
        Args:
            model_name: One of "Qwen/Qwen2.5-VL-7B-Instruct", 
                       "Qwen/Qwen2.5-VL-32B-Instruct", or 
                       "Qwen/Qwen2.5-VL-72B-Instruct"
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.db_path = "signatures.db"
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for signature storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                signature_data TEXT,
                signature_features TEXT,
                image_path TEXT,
                timestamp DATETIME,
                model_used TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def load_model(self):
        """Load the specified Qwen model"""
        print(f"Loading {self.model_name}...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            print(f"Model {self.model_name} loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_stamp_and_signature(self, image_path: str) -> Dict:
        """
        Detect stamp and signature regions in the document
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dictionary containing stamp and signature regions
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # Use VLM to detect stamp and signature regions
        prompt = """
        Analyze this document image and identify:
        1. Any stamp or seal regions (usually circular or rectangular with official markings)
        2. Signature regions (handwritten signatures, usually at the bottom)
        
        Please provide the coordinates of these regions in the format:
        - Stamp: [x1, y1, x2, y2] if present, otherwise null
        - Signature: [x1, y1, x2, y2] if present, otherwise null
        
        Focus on finding the actual signature, not the stamp if both are present.
        """
        
        if self.model is None:
            self.load_model()
        
        # Process image with VLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.processor(
            text=[text], 
            images=[image_path], 
            videos=None, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **image_inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(image_inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse response to extract coordinates
        regions = self._parse_region_coordinates(response)
        
        return {
            "stamp_region": regions.get("stamp"),
            "signature_region": regions.get("signature"),
            "full_image": image_rgb,
            "response": response
        }
    
    def _parse_region_coordinates(self, response: str) -> Dict:
        """Parse VLM response to extract region coordinates"""
        regions = {}
        
        # Look for stamp coordinates
        if "Stamp:" in response:
            stamp_match = response.split("Stamp:")[1].split("Signature:")[0] if "Signature:" in response else response.split("Stamp:")[1]
            if "null" not in stamp_match:
                coords = self._extract_coordinates(stamp_match)
                if coords:
                    regions["stamp"] = coords
        
        # Look for signature coordinates
        if "Signature:" in response:
            sig_match = response.split("Signature:")[1]
            if "null" not in sig_match:
                coords = self._extract_coordinates(sig_match)
                if coords:
                    regions["signature"] = coords
        
        return regions
    
    def _extract_coordinates(self, text: str) -> Optional[List[int]]:
        """Extract coordinate list from text"""
        import re
        # Look for pattern [x1, y1, x2, y2]
        pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        match = re.search(pattern, text)
        if match:
            return [int(x) for x in match.groups()]
        return None
    
    def extract_signature_features(self, image_path: str, signature_region: List[int] = None) -> Dict:
        """
        Extract signature features using VLM
        
        Args:
            image_path: Path to the document image
            signature_region: Optional bounding box [x1, y1, x2, y2] for signature
            
        Returns:
            Dictionary containing signature features and metadata
        """
        if self.model is None:
            self.load_model()
        
        # Load and crop image if region is specified
        image = Image.open(image_path)
        if signature_region:
            x1, y1, x2, y2 = signature_region
            image = image.crop((x1, y1, x2, y2))
        
        # Create signature analysis prompt
        prompt = """
        Analyze this signature image and provide:
        1. A detailed description of the signature characteristics
        2. Key visual features (stroke patterns, letter shapes, etc.)
        3. Signature style classification
        4. Any unique identifying features
        
        Provide a comprehensive analysis that can be used for signature comparison.
        """
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.processor(
            text=[text], 
            images=[image], 
            videos=None, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **image_inputs,
                max_new_tokens=1024,
                do_sample=False
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(image_inputs.input_ids, generated_ids)
        ]
        
        analysis = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract features for comparison
        features = self._extract_comparison_features(analysis)
        
        return {
            "analysis": analysis,
            "features": features,
            "signature_region": signature_region,
            "model_used": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_comparison_features(self, analysis: str) -> Dict:
        """Extract structured features from VLM analysis for comparison"""
        features = {
            "text_description": analysis,
            "stroke_patterns": [],
            "letter_shapes": [],
            "style_classification": "",
            "unique_features": []
        }
        
        # Simple keyword extraction (can be enhanced with more sophisticated NLP)
        analysis_lower = analysis.lower()
        
        # Extract stroke patterns
        stroke_keywords = ["curved", "straight", "loopy", "angular", "flowing", "sharp", "smooth"]
        features["stroke_patterns"] = [kw for kw in stroke_keywords if kw in analysis_lower]
        
        # Extract letter shapes
        shape_keywords = ["cursive", "print", "mixed", "capital", "lowercase", "italic", "bold"]
        features["letter_shapes"] = [kw for kw in shape_keywords if kw in analysis_lower]
        
        # Extract style classification
        if "cursive" in analysis_lower:
            features["style_classification"] = "cursive"
        elif "print" in analysis_lower:
            features["style_classification"] = "print"
        else:
            features["style_classification"] = "mixed"
        
        return features
    
    def calculate_iou_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Calculate IoU-based similarity between two signature feature sets
        
        Args:
            features1: First signature features
            features2: Second signature features
            
        Returns:
            IoU similarity score (0-1)
        """
        # Convert features to comparable vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Combine all text features
        text1 = " ".join([
            features1.get("text_description", ""),
            " ".join(features1.get("stroke_patterns", [])),
            " ".join(features1.get("letter_shapes", [])),
            features1.get("style_classification", "")
        ])
        
        text2 = " ".join([
            features2.get("text_description", ""),
            " ".join(features2.get("stroke_patterns", [])),
            " ".join(features2.get("letter_shapes", [])),
            features2.get("style_classification", "")
        ])
        
        # Create TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Convert to IoU-like metric
        iou_score = (similarity + 1) / 2  # Normalize from [-1,1] to [0,1]
        
        return iou_score
    
    def store_signature(self, user_id: str, signature_data: Dict, image_path: str):
        """Store signature data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signatures (user_id, signature_data, signature_features, image_path, timestamp, model_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            json.dumps(signature_data),
            json.dumps(signature_data.get("features", {})),
            image_path,
            signature_data.get("timestamp"),
            signature_data.get("model_used")
        ))
        
        conn.commit()
        conn.close()
    
    def find_similar_signatures(self, query_features: Dict, threshold: float = 0.7) -> List[Dict]:
        """
        Find similar signatures in database
        
        Args:
            query_features: Features of the query signature
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar signatures with similarity scores
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM signatures')
        rows = cursor.fetchall()
        conn.close()
        
        similar_signatures = []
        
        for row in rows:
            stored_features = json.loads(row[3])  # signature_features column
            similarity = self.calculate_iou_similarity(query_features, stored_features)
            
            if similarity >= threshold:
                similar_signatures.append({
                    "id": row[0],
                    "user_id": row[1],
                    "similarity": similarity,
                    "timestamp": row[5],
                    "model_used": row[6]
                })
        
        # Sort by similarity score
        similar_signatures.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar_signatures
    
    def process_document(self, image_path: str, user_id: str = None) -> Dict:
        """
        Complete document processing pipeline
        
        Args:
            image_path: Path to document image
            user_id: Optional user ID for storage
            
        Returns:
            Complete processing results
        """
        try:
            # Step 1: Detect stamp and signature regions
            regions = self.detect_stamp_and_signature(image_path)
            
            # Step 2: Extract signature features
            signature_data = self.extract_signature_features(
                image_path, 
                regions.get("signature_region")
            )
            
            # Step 3: Find similar signatures
            similar_signatures = self.find_similar_signatures(signature_data["features"])
            
            # Step 4: Determine if signature is new or matches existing
            is_new = len(similar_signatures) == 0
            best_match = similar_signatures[0] if similar_signatures else None
            
            # Step 5: Store if new or if user_id provided
            if user_id and is_new:
                self.store_signature(user_id, signature_data, image_path)
            
            return {
                "success": True,
                "regions": regions,
                "signature_data": signature_data,
                "similar_signatures": similar_signatures,
                "is_new": is_new,
                "best_match": best_match,
                "iou_score": best_match["similarity"] if best_match else 0.0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "regions": None,
                "signature_data": None,
                "similar_signatures": [],
                "is_new": True,
                "best_match": None,
                "iou_score": 0.0
            }