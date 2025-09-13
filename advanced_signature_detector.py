import torch
import cv2
import numpy as np
from PIL import Image
import json
import sqlite3
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

class AdvancedSignatureDetector:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", yolo_weights_path: str = "weights/yolov8s.pt", use_quantization: bool = True):
        """
        Advanced signature detection and comparison system using YOLOv8 for detection and Qwen VLM for analysis

        Args:
            model_name: Qwen model variant to use (default: Qwen2.5-VL-7B-Instruct for optimal performance)
            yolo_weights_path: Path to YOLOv8 weights file
            use_quantization: Whether to use 8-bit quantization to reduce memory usage
        """
        self.model_name = model_name
        self.yolo_weights_path = yolo_weights_path
        self.use_quantization = use_quantization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.yolo_model = None
        self.db_path = "advanced_signatures.db"
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.signatures_cache = None  # Cache for database signatures
        self.cache_timestamp = None   # Cache timestamp for invalidation
        self._init_database()
        self._load_yolo_model()
        
    def _init_database(self):
        """Initialize enhanced SQLite database for signature storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                signature_hash TEXT UNIQUE,
                signature_data TEXT,
                signature_features TEXT,
                embedding_vector TEXT,
                image_path TEXT,
                timestamp DATETIME,
                model_used TEXT,
                confidence_score REAL,
                signature_type TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signature_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature1_id INTEGER,
                signature2_id INTEGER,
                similarity_score REAL,
                comparison_method TEXT,
                timestamp DATETIME,
                FOREIGN KEY (signature1_id) REFERENCES signatures (id),
                FOREIGN KEY (signature2_id) REFERENCES signatures (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def _load_yolo_model(self):
        """Load YOLOv8 model for signature detection"""
        try:
            if os.path.exists(self.yolo_weights_path):
                print(f"Loading YOLOv8 model from {self.yolo_weights_path}...")
                self.yolo_model = YOLO(self.yolo_weights_path)
                print("YOLOv8 model loaded successfully!")
            else:
                print(f"YOLO weights file not found at {self.yolo_weights_path}, falling back to OpenCV detection")
                self.yolo_model = None
        except Exception as e:
            print(f"Error loading YOLO model: {e}, falling back to OpenCV detection")
            self.yolo_model = None

    def load_model(self):
        """Load the specified Qwen model with caching"""
        if self.model is not None and self.processor is not None:
            print("Model already loaded, skipping...")
            return

        print(f"Loading {self.model_name}...")
        print("This may take several minutes. Please wait...")
        try:
            # Handle different Qwen model versions
            if "Qwen2.5-VL" in self.model_name:
                # For Qwen2.5-VL models
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                print("Loading processor...")
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

                # Try loading with quantization first (if enabled and available)
                if self.use_quantization and self.device == "cuda":
                    try:
                        print("Attempting to load with 8-bit quantization...")
                        print("This will load 5 model shards. Please be patient...")
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=True
                        )
                        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            self.model_name,
                            quantization_config=quantization_config,
                            device_map="auto",
                            trust_remote_code=True
                        )
                        print("Model loaded with 8-bit quantization!")
                    except ImportError:
                        print("BitsAndBytes not available, loading without quantization...")
                        self._load_qwen25_standard()
                    except Exception as quant_error:
                        print(f"Quantization failed: {quant_error}, trying standard loading...")
                        self._load_qwen25_standard()
                else:
                    self._load_qwen25_standard()

            else:
                # For Qwen2-VL models
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                print("Loading processor...")
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                print("Loading model...")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            print(f"Model {self.model_name} loaded successfully on {self.device}!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Try restarting the application or check your internet connection.")
            raise

    def _load_qwen25_standard(self):
        """Load Qwen2.5-VL model with standard approach and memory fallbacks"""
        from transformers import Qwen2_5_VLForConditionalGeneration

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
        except RuntimeError as cuda_error:
            if "out of memory" in str(cuda_error).lower():
                print("CUDA out of memory, falling back to CPU...")
                self.device = "cpu"
                try:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        device_map=None,
                        trust_remote_code=True
                    ).to("cpu")
                except RuntimeError as cpu_error:
                    print(f"CPU loading also failed: {cpu_error}")
                    print("The Qwen2.5-VL-7B model requires more memory than available.")
                    print("Consider using a smaller model or upgrading your hardware.")
                    raise cpu_error
            else:
                raise cuda_error
    
    def detect_signature_regions_yolo(self, image_path: str) -> List[Dict]:
        """
        Advanced signature detection using YOLOv8 object detection

        Args:
            image_path: Path to the document image

        Returns:
            List of detected signature regions with metadata
        """
        if self.yolo_model is None:
            print("YOLO model not available, falling back to OpenCV detection")
            return self.detect_signature_regions_opencv(image_path)

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Run YOLO detection
            results = self.yolo_model(image, conf=0.3, verbose=False)

            regions = []

            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        # Calculate area and aspect ratio
                        w = x2 - x1
                        h = y2 - y1
                        area = w * h
                        aspect_ratio = w / h if h > 0 else 0

                        # Filter for signature-like regions based on size and aspect ratio
                        # Signatures are typically small to medium sized objects
                        image_area = image.shape[0] * image.shape[1]
                        relative_area = area / image_area

                        # Signature criteria:
                        # - Reasonable size (not too small, not dominating the image)
                        # - Aspect ratio between 1:1 and 5:1 (typical for signatures)
                        # - Reasonable confidence from YOLO
                        if (0.001 < relative_area < 0.3 and
                            0.5 < aspect_ratio < 5.0 and
                            conf > 0.3):

                            regions.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'method': 'yolo_detection',
                                'confidence': float(conf),
                                'yolo_class': cls,
                                'relative_area': relative_area
                            })

            # If no regions detected with YOLO, fall back to OpenCV
            if not regions:
                print("No signature-like regions detected with YOLO, falling back to OpenCV")
                return self.detect_signature_regions_opencv(image_path)

            # Remove duplicate regions and merge overlapping ones
            regions = self._merge_overlapping_regions(regions)

            # Sort by confidence and return top candidates
            regions.sort(key=lambda x: x['confidence'], reverse=True)

            return regions[:5]  # Return top 5 candidates

        except Exception as e:
            print(f"Error in YOLO detection: {e}, falling back to OpenCV")
            return self.detect_signature_regions_opencv(image_path)

    def detect_signature_regions_opencv(self, image_path: str) -> List[Dict]:
        """
        Fallback signature detection using OpenCV and image processing

        Args:
            image_path: Path to the document image

        Returns:
            List of detected signature regions with metadata
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to different color spaces for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Apply multiple detection methods
        regions = []

        # Method 1: Edge detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                # Signature-like regions typically have specific aspect ratios
                if 0.5 < aspect_ratio < 4.0 and area > 2000:
                    regions.append({
                        'bbox': [x, y, x+w, y+h],
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'method': 'edge_detection',
                        'confidence': min(area / 10000, 1.0)
                    })

        # Method 2: Color-based detection (look for dark regions)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                if 0.3 < aspect_ratio < 5.0:
                    regions.append({
                        'bbox': [x, y, x+w, y+h],
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'method': 'color_detection',
                        'confidence': min(area / 8000, 1.0)
                    })

        # Method 3: Template matching for signature-like patterns
        # This is a simplified approach - in practice, you'd use more sophisticated templates
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Remove duplicate regions and merge overlapping ones
        regions = self._merge_overlapping_regions(regions)

        # Sort by confidence and return top candidates
        regions.sort(key=lambda x: x['confidence'], reverse=True)

        return regions[:5]  # Return top 5 candidates
    
    def _merge_overlapping_regions(self, regions: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping regions based on IoU threshold"""
        if not regions:
            return []
        
        # Sort by confidence
        regions.sort(key=lambda x: x['confidence'], reverse=True)
        merged = []
        
        for region in regions:
            is_duplicate = False
            for merged_region in merged:
                iou = self._calculate_iou(region['bbox'], merged_region['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(region)
        
        return merged
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def extract_signature_features_advanced(self, image_path: str, signature_region: List[int] = None) -> Dict:
        """
        Advanced signature feature extraction using VLM and image analysis
        
        Args:
            image_path: Path to the document image
            signature_region: Optional bounding box [x1, y1, x2, y2] for signature
            
        Returns:
            Dictionary containing comprehensive signature features
        """
        if self.model is None:
            self.load_model()
        
        # Load and crop image if region is specified
        image = Image.open(image_path)
        if signature_region:
            x1, y1, x2, y2 = signature_region
            image = image.crop((x1, y1, x2, y2))

        # Resize large images for faster processing while maintaining quality
        max_size = 1024  # Maximum dimension
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create comprehensive signature analysis prompt
        prompt = """
        Analyze this signature image comprehensively and provide:
        
        1. DETAILED VISUAL DESCRIPTION:
           - Overall signature style and characteristics
           - Stroke patterns and pen pressure variations
           - Letter shapes and formations
           - Signature flow and directionality
           - Unique identifying features
        
        2. SIGNATURE CLASSIFICATION:
           - Signature type (cursive, print, mixed, artistic)
           - Complexity level (simple, moderate, complex)
           - Size characteristics (small, medium, large)
           - Legibility level (high, medium, low)
        
        3. COMPARATIVE FEATURES:
           - Distinctive letter formations
           - Signature baseline characteristics
           - Pen stroke patterns
           - Signature proportions
           - Any unique flourishes or decorations
        
        4. TECHNICAL ANALYSIS:
           - Signature density and ink distribution
           - Stroke thickness variations
           - Signature angle and slant
           - Overall signature geometry
        
        Provide a detailed analysis that can be used for accurate signature comparison and verification.
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
        inputs = self.processor(
            text=[text],
            images=[image],
            videos=None,
            return_tensors="pt"
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.1
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        analysis = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract comprehensive features
        features = self._extract_advanced_features(analysis, image)
        
        # Generate signature hash for uniqueness
        signature_hash = self._generate_signature_hash(image, features)
        
        return {
            "analysis": analysis,
            "features": features,
            "signature_hash": signature_hash,
            "signature_region": signature_region,
            "model_used": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "confidence_score": self._calculate_confidence_score(features)
        }
    
    def _extract_advanced_features(self, analysis: str, image: Image.Image) -> Dict:
        """Extract comprehensive structured features from VLM analysis"""
        features = {
            "text_description": analysis,
            "visual_characteristics": [],
            "signature_type": "",
            "complexity_level": "",
            "size_characteristics": "",
            "legibility_level": "",
            "stroke_patterns": [],
            "letter_shapes": [],
            "unique_features": [],
            "technical_analysis": {},
            "comparative_features": []
        }
        
        analysis_lower = analysis.lower()
        
        # Extract visual characteristics
        visual_keywords = [
            "curved", "straight", "loopy", "angular", "flowing", "sharp", "smooth",
            "bold", "light", "thick", "thin", "dense", "sparse", "elegant", "rough"
        ]
        features["visual_characteristics"] = [kw for kw in visual_keywords if kw in analysis_lower]
        
        # Extract signature type
        if "cursive" in analysis_lower:
            features["signature_type"] = "cursive"
        elif "print" in analysis_lower:
            features["signature_type"] = "print"
        elif "artistic" in analysis_lower:
            features["signature_type"] = "artistic"
        else:
            features["signature_type"] = "mixed"
        
        # Extract complexity level
        if "complex" in analysis_lower:
            features["complexity_level"] = "complex"
        elif "simple" in analysis_lower:
            features["complexity_level"] = "simple"
        else:
            features["complexity_level"] = "moderate"
        
        # Extract size characteristics
        if "large" in analysis_lower:
            features["size_characteristics"] = "large"
        elif "small" in analysis_lower:
            features["size_characteristics"] = "small"
        else:
            features["size_characteristics"] = "medium"
        
        # Extract legibility level
        if "high" in analysis_lower and "legib" in analysis_lower:
            features["legibility_level"] = "high"
        elif "low" in analysis_lower and "legib" in analysis_lower:
            features["legibility_level"] = "low"
        else:
            features["legibility_level"] = "medium"
        
        # Extract stroke patterns
        stroke_keywords = ["thick", "thin", "varying", "consistent", "pressure", "flow"]
        features["stroke_patterns"] = [kw for kw in stroke_keywords if kw in analysis_lower]
        
        # Extract letter shapes
        shape_keywords = ["capital", "lowercase", "italic", "bold", "rounded", "sharp"]
        features["letter_shapes"] = [kw for kw in shape_keywords if kw in analysis_lower]
        
        # Extract unique features
        unique_keywords = ["flourish", "decoration", "unique", "distinctive", "special"]
        features["unique_features"] = [kw for kw in unique_keywords if kw in analysis_lower]
        
        # Technical analysis
        features["technical_analysis"] = {
            "density": "high" if "dense" in analysis_lower else "low",
            "angle": "slanted" if "slant" in analysis_lower else "straight",
            "geometry": "regular" if "regular" in analysis_lower else "irregular"
        }
        
        return features
    
    def _generate_signature_hash(self, image: Image.Image, features: Dict) -> str:
        """Generate unique hash for signature based on image and features"""
        # Convert image to bytes
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Create hash from image data and key features
        hash_input = img_data + str(features.get("signature_type", "")).encode()
        return hashlib.md5(hash_input).hexdigest()
    
    def _calculate_confidence_score(self, features: Dict) -> float:
        """Calculate confidence score based on feature completeness"""
        score = 0.0
        total_features = 0
        
        # Check feature completeness
        feature_checks = [
            features.get("signature_type"),
            features.get("complexity_level"),
            features.get("size_characteristics"),
            features.get("legibility_level"),
            len(features.get("visual_characteristics", [])),
            len(features.get("stroke_patterns", [])),
            len(features.get("unique_features", []))
        ]
        
        for check in feature_checks:
            total_features += 1
            if check:
                score += 1.0
        
        return score / total_features if total_features > 0 else 0.0
    
    def calculate_advanced_similarity(self, features1: Dict, features2: Dict) -> Dict:
        """
        Calculate comprehensive similarity between two signature feature sets
        
        Args:
            features1: First signature features
            features2: Second signature features
            
        Returns:
            Dictionary containing multiple similarity metrics
        """
        similarities = {}
        
        # Text similarity using TF-IDF
        text1 = features1.get("text_description", "")
        text2 = features2.get("text_description", "")
        
        if text1 and text2:
            try:
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                similarities["text_similarity"] = float(text_similarity)
            except:
                similarities["text_similarity"] = 0.0
        else:
            similarities["text_similarity"] = 0.0
        
        # Feature-based similarity
        feature_similarities = []
        
        # Signature type similarity
        type1 = features1.get("signature_type", "")
        type2 = features2.get("signature_type", "")
        type_sim = 1.0 if type1 == type2 else 0.0
        feature_similarities.append(type_sim)
        
        # Complexity similarity
        comp1 = features1.get("complexity_level", "")
        comp2 = features2.get("complexity_level", "")
        comp_sim = 1.0 if comp1 == comp2 else 0.0
        feature_similarities.append(comp_sim)
        
        # Visual characteristics similarity
        vis1 = set(features1.get("visual_characteristics", []))
        vis2 = set(features2.get("visual_characteristics", []))
        if vis1 or vis2:
            vis_sim = len(vis1.intersection(vis2)) / len(vis1.union(vis2))
        else:
            vis_sim = 0.0
        feature_similarities.append(vis_sim)
        
        # Stroke patterns similarity
        stroke1 = set(features1.get("stroke_patterns", []))
        stroke2 = set(features2.get("stroke_patterns", []))
        if stroke1 or stroke2:
            stroke_sim = len(stroke1.intersection(stroke2)) / len(stroke1.union(stroke2))
        else:
            stroke_sim = 0.0
        feature_similarities.append(stroke_sim)
        
        # Calculate weighted average
        similarities["feature_similarity"] = np.mean(feature_similarities)
        
        # Overall similarity (weighted combination)
        similarities["overall_similarity"] = (
            0.4 * similarities["text_similarity"] + 
            0.6 * similarities["feature_similarity"]
        )
        
        return similarities
    
    def store_signature_advanced(self, user_id: str, signature_data: Dict, image_path: str):
        """Store signature data in enhanced database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Prepare embedding vector (simplified - in practice, use proper embeddings)
        embedding_vector = json.dumps(signature_data.get("features", {}))
        
        cursor.execute('''
            INSERT OR REPLACE INTO signatures 
            (user_id, signature_hash, signature_data, signature_features, embedding_vector, 
             image_path, timestamp, model_used, confidence_score, signature_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            signature_data.get("signature_hash"),
            json.dumps(signature_data),
            json.dumps(signature_data.get("features", {})),
            embedding_vector,
            image_path,
            signature_data.get("timestamp"),
            signature_data.get("model_used"),
            signature_data.get("confidence_score", 0.0),
            signature_data.get("features", {}).get("signature_type", "unknown")
        ))
        
        conn.commit()
        conn.close()
    
    def find_similar_signatures_advanced(self, query_features: Dict, threshold: float = 0.7) -> List[Dict]:
        """
        Find similar signatures using advanced similarity metrics with caching

        Args:
            query_features: Features of the query signature
            threshold: Similarity threshold (0-1)

        Returns:
            List of similar signatures with detailed similarity scores
        """
        # Check cache validity (invalidate after 5 minutes or if not loaded)
        current_time = datetime.now()
        if (self.signatures_cache is None or
            self.cache_timestamp is None or
            (current_time - self.cache_timestamp).seconds > 300):

            # Load fresh data from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM signatures')
            rows = cursor.fetchall()
            conn.close()

            # Cache the data
            self.signatures_cache = rows
            self.cache_timestamp = current_time

        similar_signatures = []

        for row in self.signatures_cache:
            stored_features = json.loads(row[4])  # signature_features column
            similarities = self.calculate_advanced_similarity(query_features, stored_features)

            if similarities["overall_similarity"] >= threshold:
                similar_signatures.append({
                    "id": row[0],
                    "user_id": row[1],
                    "signature_hash": row[2],
                    "similarities": similarities,
                    "overall_score": similarities["overall_similarity"],
                    "timestamp": row[7],
                    "model_used": row[8],
                    "confidence_score": row[9]
                })

        # Sort by overall similarity score
        similar_signatures.sort(key=lambda x: x["overall_score"], reverse=True)

        return similar_signatures
    
    def process_document_advanced(self, image_path: str, user_id: str = None) -> Dict:
        """
        Complete advanced document processing pipeline
        
        Args:
            image_path: Path to document image
            user_id: Optional user ID for storage
            
        Returns:
            Complete processing results with advanced analysis
        """
        try:
            # Step 1: Detect signature regions using YOLOv8
            regions = self.detect_signature_regions_yolo(image_path)
            
            # Step 2: Process each detected region
            signature_results = []
            
            for i, region in enumerate(regions):
                signature_data = self.extract_signature_features_advanced(
                    image_path, 
                    region['bbox']
                )
                signature_data['detection_confidence'] = region['confidence']
                signature_data['detection_method'] = region['method']
                signature_results.append(signature_data)
            
            # Step 3: Find similar signatures for each detected signature
            all_similar_signatures = []
            for signature_data in signature_results:
                similar_signatures = self.find_similar_signatures_advanced(
                    signature_data["features"]
                )
                all_similar_signatures.extend(similar_signatures)
            
            # Step 4: Determine if signatures are new or match existing
            new_signatures = []
            matched_signatures = []
            
            for signature_data in signature_results:
                if signature_data["signature_hash"] not in [s["signature_hash"] for s in all_similar_signatures]:
                    new_signatures.append(signature_data)
                else:
                    matched_signatures.append(signature_data)
            
            # Step 5: Store new signatures if user_id provided
            if user_id:
                for signature_data in new_signatures:
                    self.store_signature_advanced(user_id, signature_data, image_path)
            
            return {
                "success": True,
                "detected_regions": regions,
                "signature_results": signature_results,
                "new_signatures": new_signatures,
                "matched_signatures": matched_signatures,
                "similar_signatures": all_similar_signatures,
                "total_signatures_detected": len(signature_results),
                "new_signatures_count": len(new_signatures),
                "matched_signatures_count": len(matched_signatures)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "detected_regions": [],
                "signature_results": [],
                "new_signatures": [],
                "matched_signatures": [],
                "similar_signatures": [],
                "total_signatures_detected": 0,
                "new_signatures_count": 0,
                "matched_signatures_count": 0
            }
    
    def generate_comparison_report(self, signature1_id: int, signature2_id: int) -> Dict:
        """
        Generate detailed comparison report between two signatures
        
        Args:
            signature1_id: ID of first signature
            signature2_id: ID of second signature
            
        Returns:
            Detailed comparison report
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get signature data
        cursor.execute('SELECT * FROM signatures WHERE id = ?', (signature1_id,))
        sig1_data = cursor.fetchone()
        
        cursor.execute('SELECT * FROM signatures WHERE id = ?', (signature2_id,))
        sig2_data = cursor.fetchone()
        
        conn.close()
        
        if not sig1_data or not sig2_data:
            return {"error": "One or both signatures not found"}
        
        # Extract features
        features1 = json.loads(sig1_data[4])
        features2 = json.loads(sig2_data[4])
        
        # Calculate similarities
        similarities = self.calculate_advanced_similarity(features1, features2)
        
        # Store comparison in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO signature_comparisons 
            (signature1_id, signature2_id, similarity_score, comparison_method, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            signature1_id, 
            signature2_id, 
            similarities["overall_similarity"],
            "advanced_analysis",
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
        
        return {
            "signature1": {
                "id": sig1_data[0],
                "user_id": sig1_data[1],
                "signature_type": features1.get("signature_type", "unknown"),
                "confidence_score": sig1_data[9]
            },
            "signature2": {
                "id": sig2_data[0],
                "user_id": sig2_data[1],
                "signature_type": features2.get("signature_type", "unknown"),
                "confidence_score": sig2_data[9]
            },
            "similarities": similarities,
            "verdict": "MATCH" if similarities["overall_similarity"] > 0.8 else "NO_MATCH",
            "confidence": similarities["overall_similarity"]
        }
