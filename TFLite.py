import tensorflow as tf

# 학습된 모델 불러오기
model = tf.keras.models.load_model("sixcnn_model.h5")

# TFLite 변환기 사용
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 최적화 옵션 설정 (파라미터 수 줄이기)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 변환 실행
tflite_model = converter.convert()

# 파일로 저장
with open("model.tflite", "wb") as f:
    f.write(tflite_model)