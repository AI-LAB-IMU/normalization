import tensorflow as tf

# 모델 불러오기
model = tf.keras.models.load_model("sixcnn_model.h5")

# 변환기 초기화
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 하위 연산자 버전 강제
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# 변환 수행
tflite_model = converter.convert()

# 저장
with open("model_2.tflite", "wb") as f:
    f.write(tflite_model)
