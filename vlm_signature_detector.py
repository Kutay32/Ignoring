import torch
import cv2
import numpy as np
from PIL import Image
import json
import sqlite3
from typing import List, Dict, Tuple, Optional, Union
import os
from datetime import datetime
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")

class VLMSignatureDetector:
    """
    VLM-style signature detection module that combines YOLOv8s.pt for object detection
    with Vision-Language Model capabilities for signature analysis and comparison.
    """
    
    def __init__(self, 
                 yolo_model_path: str = "weights/yolov8s.pt",
                 vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
                 confidence_threshold: float = 0.5):
        """
        Initialize the VLM Signature Detector
        
        Args:
            yolo_model_path: Path to YOLOv8s.pt model weights
            vlm_model_name: VLM model name for signature analysis
            confidence_threshold: YOLO detection confidence threshold
        """
        self.yolo_model_path = yolo_model_path
        self.vlm_model_name = vlm_model_name
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self.yolo_model = None
        self.vlm_model = None
        self.vlm_processor = None
        
        # Database and analysis tools
        self.db_path = "vlm_signatures.db"
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize components
        self._init_database()
        self._load_yolo_model()
        self._load_vlm_model()
    
    def _init_database(self):
        """Initialize SQLite database for VLM signature storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main signatures table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vlm_signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                signature_hash TEXT UNIQUE,
                signature_data TEXT,
                vlm_analysis TEXT,
                yolo_detection TEXT,
                signature_features TEXT,
                embedding_vector TEXT,
                image_path TEXT,
                timestamp DATETIME,
                yolo_confidence REAL,
                vlm_confidence REAL,
                signature_type TEXT,
                structure_type TEXT
            )
        ''')
        
        # Signature comparisons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signature_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature1_id INTEGER,
                signature2_id INTEGER,
                yolo_similarity REAL,
                vlm_similarity REAL,
                overall_similarity REAL,
                comparison_method TEXT,
                timestamp DATETIME,
                FOREIGN KEY (signature1_id) REFERENCES vlm_signatures (id),
                FOREIGN KEY (signature2_id) REFERENCES vlm_signatures (id)
            )
        ''')
        
        # VLM structure analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS structure_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature_id INTEGER,
                structure_type TEXT,
                structure_features TEXT,
                vlm_confidence REAL,
                analysis_timestamp DATETIME,
                FOREIGN KEY (signature_id) REFERENCES vlm_signatures (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_yolo_model(self):
        """Load YOLOv8s model for object detection"""
        try:
            print(f"Loading YOLOv8s model from {self.yolo_model_path}...")
            self.yolo_model = YOLO(self.yolo_model_path)
            print("YOLOv8s model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLOv8s model: {e}")
            raise
    
    def _load_vlm_model(self):
        """Load VLM model for signature analysis"""
        try:
            print(f"Loading VLM model: {self.vlm_model_name}...")
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            self.vlm_processor = AutoProcessor.from_pretrained(
                self.vlm_model_name, 
                trust_remote_code=True
            )
            self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.vlm_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            print(f"VLM model {self.vlm_model_name} loaded successfully!")
        except Exception as e:
            print(f"Error loading VLM model: {e}")
            # Fallback to smaller model if available
            if "7B" in self.vlm_model_name or "32B" in self.vlm_model_name:
                print("Trying with smaller 2B model...")
                self.vlm_model_name = "Qwen/Qwen2-VL-2B-Instruct"
                self._load_vlm_model()
            else:
                raise
    
    def detect_signature_regions_yolo(self, image_path: str) -> List[Dict]:
        """
        Detect signature regions using YOLOv8s
        
        Args:
            image_path: Path to the document image
            
        Returns:
            List of detected signature regions with YOLO metadata
        """
        try:
            # Run YOLO detection
            results = self.yolo_model(image_path, conf=self.confidence_threshold)
            
            signature_regions = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Filter for potential signature regions based on size and aspect ratio
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # Signature-like regions typically have specific characteristics
                        if (area > 1000 and  # Minimum area
                            0.2 < aspect_ratio < 5.0 and  # Reasonable aspect ratio
                            conf > self.confidence_threshold):
                            
                            signature_regions.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'class_id': int(class_id),
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'method': 'yolo_detection',
                                'width': width,
                                'height': height
                            })
            
            # Sort by confidence and return top candidates
            signature_regions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return signature_regions[:10]  # Return top 10 candidates
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def analyze_signature_structure_vlm(self, image_path: str, signature_region: List[int] = None) -> Dict:
        """
        Analyze signature structure using VLM for VLM-style analysis
        
        Args:
            image_path: Path to the document image
            signature_region: Optional bounding box [x1, y1, x2, y2] for signature
            
        Returns:
            Dictionary containing VLM analysis of signature structure
        """
        try:
            # Load and crop image if region is specified
            image = Image.open(image_path)
            if signature_region:
                x1, y1, x2, y2 = signature_region
                image = image.crop((x1, y1, x2, y2))
            
            # Create comprehensive VLM analysis prompt for signature structure
            prompt = """
            Analyze this signature image comprehensively for VLM-style structure detection:
            
            1. STRUCTURAL ANALYSIS:
               - Overall signature architecture and layout
               - Hierarchical structure of letters and components
               - Spatial relationships between signature elements
               - Signature flow and directional patterns
            
            2. VISUAL LANGUAGE MODELING FEATURES:
               - Token-level characteristics (individual letter structures)
               - Sequence patterns (letter combinations and transitions)
               - Contextual relationships between signature components
               - Semantic structure and meaning representation
            
            3. SIGNATURE COMPONENTS:
               - Individual letter formations and their characteristics
               - Connecting strokes and transitions
               - Signature baseline and alignment patterns
               - Unique structural elements and flourishes
            
            4. COMPARATIVE STRUCTURE FEATURES:
               - Distinctive structural patterns
               - Signature geometry and proportions
               - Stroke density and distribution patterns
               - Structural complexity indicators
            
            5. VLM-SPECIFIC METRICS:
               - Information density and entropy
               - Structural consistency measures
               - Pattern recognition features
               - Signature encoding characteristics
            
            Provide a detailed structural analysis that can be used for accurate VLM-style signature comparison and verification.
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
            
            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.vlm_processor(
                text=[text], 
                images=[image], 
                videos=None, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                generated_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=0.1
                )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            vlm_analysis = self.vlm_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            # Extract structured features from VLM analysis
            structure_features = self._extract_vlm_structure_features(vlm_analysis, image)
            
            return {
                "vlm_analysis": vlm_analysis,
                "structure_features": structure_features,
                "signature_region": signature_region,
                "vlm_confidence": self._calculate_vlm_confidence(structure_features),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in VLM analysis: {e}")
            return {
                "vlm_analysis": "",
                "structure_features": {},
                "signature_region": signature_region,
                "vlm_confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _extract_vlm_structure_features(self, analysis: str, image: Image.Image) -> Dict:
        """Extract structured features from VLM analysis for VLM-style comparison"""
        features = {
            "text_analysis": analysis,
            "structural_components": [],
            "vlm_tokens": [],
            "spatial_relationships": [],
            "signature_architecture": "",
            "complexity_metrics": {},
            "pattern_features": [],
            "semantic_structure": [],
            "comparative_features": []
        }
        
        analysis_lower = analysis.lower()
        
        # Extract structural components
        structural_keywords = [
            "hierarchical", "layered", "modular", "sequential", "parallel",
            "connected", "isolated", "grouped", "distributed", "centralized"
        ]
        features["structural_components"] = [
            kw for kw in structural_keywords if kw in analysis_lower
        ]
        
        # Extract VLM token characteristics
        token_keywords = [
            "token", "sequence", "pattern", "encoding", "representation",
            "embedding", "contextual", "semantic", "syntactic"
        ]
        features["vlm_tokens"] = [
            kw for kw in token_keywords if kw in analysis_lower
        ]
        
        # Extract spatial relationships
        spatial_keywords = [
            "above", "below", "left", "right", "center", "peripheral",
            "adjacent", "overlapping", "separated", "aligned", "offset"
        ]
        features["spatial_relationships"] = [
            kw for kw in spatial_keywords if kw in analysis_lower
        ]
        
        # Determine signature architecture
        if "hierarchical" in analysis_lower:
            features["signature_architecture"] = "hierarchical"
        elif "modular" in analysis_lower:
            features["signature_architecture"] = "modular"
        elif "sequential" in analysis_lower:
            features["signature_architecture"] = "sequential"
        else:
            features["signature_architecture"] = "mixed"
        
        # Extract complexity metrics
        complexity_indicators = {
            "high_complexity": ["complex", "intricate", "sophisticated", "detailed"],
            "medium_complexity": ["moderate", "balanced", "structured"],
            "low_complexity": ["simple", "basic", "minimal", "straightforward"]
        }
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in analysis_lower for indicator in indicators):
                features["complexity_metrics"]["level"] = level
                break
        
        # Extract pattern features
        pattern_keywords = [
            "repetitive", "unique", "consistent", "variable", "rhythmic",
            "irregular", "symmetric", "asymmetric", "flowing", "angular"
        ]
        features["pattern_features"] = [
            kw for kw in pattern_keywords if kw in analysis_lower
        ]
        
        # Extract semantic structure
        semantic_keywords = [
            "meaningful", "symbolic", "representational", "abstract", "concrete",
            "expressive", "functional", "decorative", "informative"
        ]
        features["semantic_structure"] = [
            kw for kw in semantic_keywords if kw in analysis_lower
        ]
        
        return features
    
    def _calculate_vlm_confidence(self, features: Dict) -> float:
        """Calculate confidence score based on VLM feature completeness"""
        score = 0.0
        total_features = 0
        
        # Check feature completeness
        feature_checks = [
            features.get("signature_architecture"),
            features.get("complexity_metrics", {}).get("level"),
            len(features.get("structural_components", [])),
            len(features.get("vlm_tokens", [])),
            len(features.get("spatial_relationships", [])),
            len(features.get("pattern_features", [])),
            len(features.get("semantic_structure", []))
        ]
        
        for check in feature_checks:
            total_features += 1
            if check:
                score += 1.0
        
        return score / total_features if total_features > 0 else 0.0
    
    def calculate_vlm_similarity(self, features1: Dict, features2: Dict) -> Dict:
        """
        Calculate VLM-style similarity between two signature feature sets
        
        Args:
            features1: First signature VLM features
            features2: Second signature VLM features
            
        Returns:
            Dictionary containing VLM similarity metrics
        """
        similarities = {}
        
        # Text similarity using TF-IDF
        text1 = features1.get("text_analysis", "")
        text2 = features2.get("text_analysis", "")
        
        if text1 and text2:
            try:
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                similarities["text_similarity"] = float(text_similarity)
            except:
                similarities["text_similarity"] = 0.0
        else:
            similarities["text_similarity"] = 0.0
        
        # Structural similarity
        struct1 = set(features1.get("structural_components", []))
        struct2 = set(features2.get("structural_components", []))
        if struct1 or struct2:
            struct_sim = len(struct1.intersection(struct2)) / len(struct1.union(struct2))
        else:
            struct_sim = 0.0
        similarities["structural_similarity"] = struct_sim
        
        # VLM token similarity
        tokens1 = set(features1.get("vlm_tokens", []))
        tokens2 = set(features2.get("vlm_tokens", []))
        if tokens1 or tokens2:
            token_sim = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
        else:
            token_sim = 0.0
        similarities["vlm_token_similarity"] = token_sim
        
        # Pattern similarity
        patterns1 = set(features1.get("pattern_features", []))
        patterns2 = set(features2.get("pattern_features", []))
        if patterns1 or patterns2:
            pattern_sim = len(patterns1.intersection(patterns2)) / len(patterns1.union(patterns2))
        else:
            pattern_sim = 0.0
        similarities["pattern_similarity"] = pattern_sim
        
        # Architecture similarity
        arch1 = features1.get("signature_architecture", "")
        arch2 = features2.get("signature_architecture", "")
        arch_sim = 1.0 if arch1 == arch2 else 0.0
        similarities["architecture_similarity"] = arch_sim
        
        # Overall VLM similarity (weighted combination)
        similarities["overall_vlm_similarity"] = (
            0.3 * similarities["text_similarity"] +
            0.25 * similarities["structural_similarity"] +
            0.2 * similarities["vlm_token_similarity"] +
            0.15 * similarities["pattern_similarity"] +
            0.1 * similarities["architecture_similarity"]
        )
        
        return similarities
    
    def process_document_vlm(self, image_path: str, user_id: str = None) -> Dict:
        """
        Complete VLM-style document processing pipeline
        
        Args:
            image_path: Path to document image
            user_id: Optional user ID for storage
            
        Returns:
            Complete VLM processing results
        """
        try:
            # Step 1: Detect signature regions using YOLOv8s
            yolo_regions = self.detect_signature_regions_yolo(image_path)
            
            # Step 2: Analyze each detected region with VLM
            vlm_results = []
            
            for i, region in enumerate(yolo_regions):
                vlm_analysis = self.analyze_signature_structure_vlm(
                    image_path, 
                    region['bbox']
                )
                
                # Combine YOLO and VLM results
                combined_result = {
                    **region,
                    **vlm_analysis,
                    "combined_confidence": (
                        region['confidence'] * 0.4 + 
                        vlm_analysis['vlm_confidence'] * 0.6
                    )
                }
                vlm_results.append(combined_result)
            
            # Step 3: Find similar signatures using VLM features
            all_similar_signatures = []
            for result in vlm_results:
                similar_signatures = self.find_similar_signatures_vlm(
                    result["structure_features"]
                )
                all_similar_signatures.extend(similar_signatures)
            
            # Step 4: Store new signatures if user_id provided
            stored_signatures = []
            if user_id:
                for result in vlm_results:
                    signature_id = self.store_signature_vlm(user_id, result, image_path)
                    if signature_id:
                        stored_signatures.append(signature_id)
            
            return {
                "success": True,
                "yolo_detections": yolo_regions,
                "vlm_analyses": vlm_results,
                "similar_signatures": all_similar_signatures,
                "stored_signatures": stored_signatures,
                "total_detections": len(yolo_regions),
                "total_vlm_analyses": len(vlm_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "yolo_detections": [],
                "vlm_analyses": [],
                "similar_signatures": [],
                "stored_signatures": [],
                "total_detections": 0,
                "total_vlm_analyses": 0
            }
    
    def store_signature_vlm(self, user_id: str, signature_data: Dict, image_path: str) -> Optional[int]:
        """Store VLM signature data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate signature hash
            signature_hash = self._generate_signature_hash(signature_data)
            
            # Prepare embedding vector
            embedding_vector = json.dumps(signature_data.get("structure_features", {}))
            
            cursor.execute('''
                INSERT OR REPLACE INTO vlm_signatures 
                (user_id, signature_hash, signature_data, vlm_analysis, yolo_detection,
                 signature_features, embedding_vector, image_path, timestamp, 
                 yolo_confidence, vlm_confidence, signature_type, structure_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                signature_hash,
                json.dumps(signature_data),
                signature_data.get("vlm_analysis", ""),
                json.dumps({
                    "bbox": signature_data.get("bbox"),
                    "confidence": signature_data.get("confidence"),
                    "method": signature_data.get("method")
                }),
                json.dumps(signature_data.get("structure_features", {})),
                embedding_vector,
                image_path,
                signature_data.get("timestamp"),
                signature_data.get("confidence", 0.0),
                signature_data.get("vlm_confidence", 0.0),
                signature_data.get("structure_features", {}).get("signature_architecture", "unknown"),
                signature_data.get("structure_features", {}).get("signature_architecture", "unknown")
            ))
            
            signature_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return signature_id
            
        except Exception as e:
            print(f"Error storing signature: {e}")
            return None
    
    def find_similar_signatures_vlm(self, query_features: Dict, threshold: float = 0.7) -> List[Dict]:
        """
        Find similar signatures using VLM features
        
        Args:
            query_features: VLM features of the query signature
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar signatures with VLM similarity scores
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM vlm_signatures')
            rows = cursor.fetchall()
            conn.close()
            
            similar_signatures = []
            
            for row in rows:
                stored_features = json.loads(row[6])  # signature_features column
                similarities = self.calculate_vlm_similarity(query_features, stored_features)
                
                if similarities["overall_vlm_similarity"] >= threshold:
                    similar_signatures.append({
                        "id": row[0],
                        "user_id": row[1],
                        "signature_hash": row[2],
                        "similarities": similarities,
                        "overall_score": similarities["overall_vlm_similarity"],
                        "timestamp": row[9],
                        "yolo_confidence": row[10],
                        "vlm_confidence": row[11],
                        "signature_type": row[12]
                    })
            
            # Sort by overall similarity score
            similar_signatures.sort(key=lambda x: x["overall_score"], reverse=True)
            
            return similar_signatures
            
        except Exception as e:
            print(f"Error finding similar signatures: {e}")
            return []
    
    def _generate_signature_hash(self, signature_data: Dict) -> str:
        """Generate unique hash for signature based on VLM features"""
        # Create hash from VLM analysis and key features
        hash_input = (
            signature_data.get("vlm_analysis", "") +
            str(signature_data.get("structure_features", {}).get("signature_architecture", ""))
        ).encode()
        return hashlib.md5(hash_input).hexdigest()
    
    def compare_signatures_vlm(self, signature1_id: int, signature2_id: int) -> Dict:
        """
        Compare two signatures using VLM analysis
        
        Args:
            signature1_id: ID of first signature
            signature2_id: ID of second signature
            
        Returns:
            Detailed VLM comparison results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get signature data
            cursor.execute('SELECT * FROM vlm_signatures WHERE id = ?', (signature1_id,))
            sig1_data = cursor.fetchone()
            
            cursor.execute('SELECT * FROM vlm_signatures WHERE id = ?', (signature2_id,))
            sig2_data = cursor.fetchone()
            
            conn.close()
            
            if not sig1_data or not sig2_data:
                return {"error": "One or both signatures not found"}
            
            # Extract VLM features
            features1 = json.loads(sig1_data[6])
            features2 = json.loads(sig2_data[6])
            
            # Calculate VLM similarities
            similarities = self.calculate_vlm_similarity(features1, features2)
            
            # Store comparison in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signature_comparisons 
                (signature1_id, signature2_id, yolo_similarity, vlm_similarity, 
                 overall_similarity, comparison_method, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                signature1_id, 
                signature2_id, 
                0.0,  # YOLO similarity not calculated in this method
                similarities["overall_vlm_similarity"],
                similarities["overall_vlm_similarity"],
                "vlm_analysis",
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
            
            return {
                "signature1": {
                    "id": sig1_data[0],
                    "user_id": sig1_data[1],
                    "signature_type": sig1_data[12],
                    "vlm_confidence": sig1_data[11]
                },
                "signature2": {
                    "id": sig2_data[0],
                    "user_id": sig2_data[1],
                    "signature_type": sig2_data[12],
                    "vlm_confidence": sig2_data[11]
                },
                "similarities": similarities,
                "verdict": "MATCH" if similarities["overall_vlm_similarity"] > 0.8 else "NO_MATCH",
                "confidence": similarities["overall_vlm_similarity"]
            }
            
        except Exception as e:
            return {"error": f"Comparison failed: {str(e)}"}
    
    def get_signature_statistics(self) -> Dict:
        """Get statistics about stored signatures"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total signatures
            cursor.execute('SELECT COUNT(*) FROM vlm_signatures')
            total_signatures = cursor.fetchone()[0]
            
            # Signatures by type
            cursor.execute('SELECT signature_type, COUNT(*) FROM vlm_signatures GROUP BY signature_type')
            signatures_by_type = dict(cursor.fetchall())
            
            # Average confidence scores
            cursor.execute('SELECT AVG(yolo_confidence), AVG(vlm_confidence) FROM vlm_signatures')
            avg_confidences = cursor.fetchone()
            
            # Recent signatures
            cursor.execute('SELECT COUNT(*) FROM vlm_signatures WHERE timestamp > datetime("now", "-7 days")')
            recent_signatures = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_signatures": total_signatures,
                "signatures_by_type": signatures_by_type,
                "average_yolo_confidence": avg_confidences[0] or 0.0,
                "average_vlm_confidence": avg_confidences[1] or 0.0,
                "recent_signatures_7_days": recent_signatures
            }
            
        except Exception as e:
            return {"error": f"Failed to get statistics: {str(e)}"}