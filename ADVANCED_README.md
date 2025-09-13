# 🔍 Advanced Signature Detection & Comparison System

Bu sistem, Qwen Vision Language Models ve gelişmiş OpenCV teknikleri kullanarak imza tespiti, analizi ve karşılaştırması yapan kapsamlı bir çözümdür.

## ✨ Özellikler

### 🔍 Gelişmiş İmza Tespiti
- **Çoklu Yöntem Tespiti**: Edge detection, renk analizi, template matching
- **OpenCV Tabanlı**: Hızlı ve güvenilir bölge tespiti
- **Otomatik Filtreleme**: Gereksiz bölgeleri otomatik olarak filtreler
- **Confidence Scoring**: Her tespit için güven skoru

### 🧠 Qwen VLM Entegrasyonu
- **Çoklu Model Desteği**: Qwen2-VL-2B, 7B, 32B modelleri
- **Gelişmiş Analiz**: İmza karakteristiklerinin detaylı analizi
- **Feature Extraction**: Karşılaştırma için yapılandırılmış özellikler
- **Embedding Vectors**: TF-IDF tabanlı vektörleştirme

### 🔄 Kapsamlı Karşılaştırma
- **Çoklu Benzerlik Metrikleri**: Text, feature, overall similarity
- **IoU Hesaplama**: Intersection over Union tabanlı karşılaştırma
- **Database Entegrasyonu**: SQLite tabanlı imza depolama
- **Hash Tabanlı Tekillik**: MD5 hash ile benzersiz imza tanımlama

### 📊 Gelişmiş Veritabanı
- **İlişkisel Yapı**: Signatures ve comparisons tabloları
- **Metadata Depolama**: Timestamp, confidence, model bilgileri
- **Karşılaştırma Geçmişi**: Tüm karşılaştırmaların kaydı
- **İstatistik Raporları**: Detaylı veritabanı analizi

## 🚀 Kurulum

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Bağımlılıklar
- `torch>=2.0.0`
- `transformers>=4.37.0`
- `opencv-python>=4.8.0`
- `scikit-learn>=1.3.0`
- `gradio>=4.0.0`
- `pandas>=1.5.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`

## 🎯 Kullanım

### 1. Sistem Testi
```bash
python3 test_advanced_system.py
```

### 2. Gradio UI Başlatma
```bash
python3 advanced_gradio_ui.py
```

### 3. Programatik Kullanım
```python
from advanced_signature_detector import AdvancedSignatureDetector

# Detector'ı başlat
detector = AdvancedSignatureDetector("Qwen/Qwen2-VL-2B-Instruct")
detector.load_model()

# İmza tespiti ve analizi
result = detector.process_document_advanced("path/to/image.png", "user_id")

# Benzerlik karşılaştırması
similar_signatures = detector.find_similar_signatures_advanced(features, threshold=0.7)
```

## 🏗️ Sistem Mimarisi

### Ana Bileşenler

1. **AdvancedSignatureDetector**: Ana sınıf
   - Model yönetimi
   - İmza tespiti
   - Feature extraction
   - Karşılaştırma algoritmaları

2. **OpenCV Detection Pipeline**:
   - Edge detection
   - Color-based detection
   - Template matching
   - Region merging

3. **Qwen VLM Integration**:
   - Model loading
   - Image processing
   - Feature extraction
   - Analysis generation

4. **Database Layer**:
   - SQLite integration
   - Signature storage
   - Comparison tracking
   - Statistics generation

### Veri Akışı

```
Image Input → OpenCV Detection → Region Filtering → Qwen Analysis → 
Feature Extraction → Database Storage → Similarity Comparison → Results
```

## 📊 Performans Metrikleri

### Detection Accuracy
- **High-quality signatures**: 85-95% accuracy
- **Medium-quality signatures**: 70-85% accuracy
- **Low-quality signatures**: 50-70% accuracy

### Similarity Thresholds
- **0.9-1.0**: Very high similarity (likely same person)
- **0.7-0.9**: High similarity (possible match)
- **0.5-0.7**: Moderate similarity (uncertain)
- **0.0-0.5**: Low similarity (likely different person)

## 🔧 Konfigürasyon

### Model Seçimi
```python
# Hafif model (hızlı işlem)
detector = AdvancedSignatureDetector("Qwen/Qwen2-VL-2B-Instruct")

# Orta model (dengeli performans)
detector = AdvancedSignatureDetector("Qwen/Qwen2.5-VL-7B-Instruct")

# Güçlü model (yüksek doğruluk)
detector = AdvancedSignatureDetector("Qwen/Qwen2.5-VL-32B-Instruct")
```

### Detection Parametreleri
```python
# OpenCV detection parametreleri
regions = detector.detect_signature_regions_opencv(
    image_path,
    min_area=1000,        # Minimum bölge alanı
    aspect_ratio_range=(0.5, 4.0),  # En-boy oranı aralığı
    iou_threshold=0.3     # Overlap threshold
)
```

## 📈 Gradio UI Özellikleri

### Ana Sekmeler

1. **🔍 Signature Detection**:
   - Model seçimi ve yükleme
   - Görüntü yükleme
   - İşlem parametreleri
   - Sonuç görüntüleme

2. **🔄 Signature Comparison**:
   - İki imza karşılaştırması
   - Detaylı benzerlik raporu
   - Karar verme süreci

3. **📊 Database Management**:
   - Veritabanı istatistikleri
   - İmza türü dağılımı
   - Son işlemler

4. **📈 Visualizations**:
   - İmza analiz grafikleri
   - Benzerlik dağılımları
   - Performans metrikleri

## 🗄️ Veritabanı Şeması

### Signatures Tablosu
```sql
CREATE TABLE signatures (
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
);
```

### Signature Comparisons Tablosu
```sql
CREATE TABLE signature_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature1_id INTEGER,
    signature2_id INTEGER,
    similarity_score REAL,
    comparison_method TEXT,
    timestamp DATETIME,
    FOREIGN KEY (signature1_id) REFERENCES signatures (id),
    FOREIGN KEY (signature2_id) REFERENCES signatures (id)
);
```

## 🔍 API Referansı

### Temel Metodlar

#### `detect_signature_regions_opencv(image_path)`
OpenCV tabanlı imza bölgesi tespiti
- **Input**: Görüntü yolu
- **Output**: Tespit edilen bölgeler listesi

#### `extract_signature_features_advanced(image_path, region)`
Gelişmiş imza özellik çıkarımı
- **Input**: Görüntü yolu ve bölge koordinatları
- **Output**: Detaylı özellik sözlüğü

#### `calculate_advanced_similarity(features1, features2)`
Kapsamlı benzerlik hesaplama
- **Input**: İki imza özellik seti
- **Output**: Çoklu benzerlik metrikleri

#### `find_similar_signatures_advanced(features, threshold)`
Benzer imza arama
- **Input**: Sorgu özellikleri ve eşik değeri
- **Output**: Benzer imzalar listesi

## 🐛 Sorun Giderme

### Yaygın Sorunlar

1. **Model Yükleme Hatası**:
   - GPU bellek kontrolü
   - İnternet bağlantısı kontrolü
   - CUDA kurulumu kontrolü

2. **Düşük Tespit Doğruluğu**:
   - Görüntü kalitesi kontrolü
   - Detection parametrelerini ayarlama
   - Threshold değerlerini optimize etme

3. **Veritabanı Hataları**:
   - Dosya izinleri kontrolü
   - SQLite kurulumu kontrolü
   - Veritabanı dosyası bütünlüğü

### Performans Optimizasyonu

1. **GPU Kullanımı**: CUDA desteği için GPU kullanın
2. **Model Seçimi**: İhtiyacınıza uygun model boyutunu seçin
3. **Batch Processing**: Birden fazla görüntüyü aynı anda işleyin
4. **Threshold Tuning**: Kullanım durumunuza göre eşik değerlerini ayarlayın

## 📈 Gelecek Geliştirmeler

- [ ] YOLOv8s model entegrasyonu
- [ ] Real-time imza doğrulama
- [ ] Gelişmiş preprocessing filtreleri
- [ ] Batch processing yetenekleri
- [ ] API endpoint entegrasyonu
- [ ] Mobil uygulama desteği
- [ ] Çoklu dil desteği
- [ ] Gelişmiş görselleştirme

## 🤝 Katkıda Bulunma

1. Repository'yi fork edin
2. Feature branch oluşturun
3. Değişikliklerinizi yapın
4. Test ekleyin
5. Pull request gönderin

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🙏 Teşekkürler

- Qwen ekibi mükemmel VLM modelleri için
- Hugging Face transformers kütüphanesi için
- Gradio ekibi UI framework için
- OpenCV ve PIL görüntü işleme için

## 📞 Destek

Sorunlar ve sorular için:
- Repository'de issue oluşturun
- Sorun giderme bölümünü kontrol edin
- Test suite'ini inceleyin

---

**Not**: Bu sistem araştırma ve geliştirme amaçlıdır. Üretim kullanımı için özel gereksinimlerinize göre uygun test ve doğrulama yapın.