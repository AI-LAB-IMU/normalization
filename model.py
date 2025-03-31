import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# 데이터 로드
df = pd.read_csv("cleaned_data_6666.csv")
print(df.columns)

# 특성과 라벨 분리
X = df.drop(columns=['index', 'y_class'])
y = df['y_class'].values

# 훈련/검증 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# MinMax 스케일링 적용
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일링 정보 출력 (최소값, 최대값)
print("Min values:", scaler.data_min_)
print("Max values:", scaler.data_max_)

# Conv1D 입력 형태로 reshape
X_train_scaled = X_train_scaled.reshape(-1, X_train.shape[1], 1)
X_test_scaled = X_test_scaled.reshape(-1, X_test.shape[1], 1)

# 라벨을 one-hot 인코딩
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

# 모델 정의
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 클래스 수에 맞게 조정
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(X_train_scaled, y_train_cat, validation_data=(X_test_scaled, y_test_cat), epochs=10, batch_size=32)

# 모델 저장
model.save("sixcnn_model.h5")

# 테스트 데이터로 예측 및 평가
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n정확도:", accuracy_score(y_test, y_pred))
print("\n분류 리포트:\n", classification_report(y_test, y_pred))
