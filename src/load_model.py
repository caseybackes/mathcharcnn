import tensorflow as tf 
def load_model(model_name):
    return tf.keras.models.load_model(model_name)
