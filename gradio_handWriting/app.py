import gradio as gr
import tensorflow as tf
import cv2

model = tf.keras.models.load_model("model.h5")

def recognize_digit(image):
    if image is not None:
        # Resize the image to (28, 28)
        image = cv2.resize(image, (28, 28))
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Reshape the image to (1, 28, 28, 1)
        image = image.reshape((1, 28, 28, 1)).astype("float32") / 255.0
        prediction = model.predict(image).tolist()[0]
        return {str(i): prediction[i] for i in range(10)}
    else:
        return ""
    
ifrace = gr.Interface(
    fn=recognize_digit,
    # inputs= gr.inputs.Image(image_mode="RGB", height=28, width=28),
    # outputs=gr.outputs.Label(num_top_classes=3),
    inputs= gr.Image( image_mode = "RGB"),
    outputs=gr.Label(num_top_classes=3),
    live=True
)

ifrace.launch()

