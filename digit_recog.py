# Gerekli kütüphaneleri yükleyin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Veri Yükleme
# Kaggle'dan indirilen veri setlerini aynı dizine koymalısınız.
train_data = pd.read_csv('train.csv')  # Eğitim veri seti
test_data = pd.read_csv('test.csv')    # Test veri seti

# Veri setlerinin boyutlarını inceleyin
print(f"Train Data Shape: {train_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

# İlk birkaç satırı gözlemleyin
print(train_data.head())

# 2. Veri Keşfi ve Hazırlığı
# Eğitim setinde etiketleri (label) ve özellikleri (piksel değerleri) ayırın
X = train_data.drop(columns=['label'])  # Görüntü piksel değerleri
y = train_data['label']                 # Etiketler (sınıflar)

# Piksel değerlerini normalize edin (0-255 -> 0-1 aralığı)
X = X / 255.0
test_data = test_data / 255.0

# Eğitim ve doğrulama verilerini ayırın
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve doğrulama veri boyutlarını kontrol edin
print(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
print(f"X_val Shape: {X_val.shape}, y_val Shape: {y_val.shape}")

# 3. Model Eğitimi
# Logistic Regression kullanarak modeli oluşturun
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)

# 4. Model Değerlendirmesi
# Doğrulama setinde tahmin yapma
y_pred = model.predict(X_val)

# Doğruluk oranı
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Karışıklık Matrisi
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Sınıflandırma Raporu
print("Classification Report:\n", classification_report(y_val, y_pred))

# 5. Test Veri Setinde Tahmin
test_predictions = model.predict(test_data)

# İlk 10 tahmini ekrana yazdırma
print("First 10 Predictions on Test Data:", test_predictions[:10])

# 6. Kaggle İçin Tahmin Sonuçlarını Kaydetme
# Tahmin sonuçlarını Kaggle formatında CSV olarak kaydedin
submission = pd.DataFrame({'ImageId': range(1, len(test_predictions) + 1), 'Label': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")
