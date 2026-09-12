# coding: utf-8

"""
Flower Image Classifier Client
Gradio interface calling TensorFlow Serving REST API without TensorFlow dependency.
"""

import json
import requests
import numpy as np
from PIL import Image
import gradio as gr

# Model and server configuration
TENSORFLOW_SERVING_ENDPOINT = "http://localhost:8501/v1/models/flower_classifier:predict"
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7862

# Expected dimensions by the model during training
WIDTH = 150
HEIGHT = 150


def predict(image: Image.Image):
    """
    Processes the image received from Gradio (PIL), converts it to a normalized tensor (1, 150, 150, 3)
    and queries the TensorFlow Serving REST endpoint.
    """
    if image is None:
        return json.dumps({"error": "No image provided."})

    try:
        # 1. Force RGB format (removes Alpha channel from transparent PNGs or Grayscale mode)
        image = image.convert("RGB")

        # 2. Resize to strict dimensions expected by the model
        image = image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)

        # 3. Convert to float NumPy array
        img_array = np.array(image, dtype=np.float32)

        # 4. Normalize to [0, 1] if not already included directly in the SavedModel
        if img_array.max() > 1.0:
            img_array = img_array / 255.0

        # 5. Add batch dimension -> shape (1, 150, 150, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        # 6. Prepare JSON payload for TensorFlow Serving
        payload = {
            "signature_name": "serving_default",
            # "instances": img_batch.tolist(),
            "inputs": {
                "args_0": img_batch.tolist()  # shape must be (1, 150, 150, 3)
            },
        }

        # 7. HTTP POST request to TF Serving container
        response = requests.post(
            TENSORFLOW_SERVING_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )

        # If TF Serving returns an error (400, 404, 500...), show the exact error message
        if response.status_code != 200:
            print(f"Status: {response.status_code}")
            print(f"Error details: {response.text}")
            return json.dumps({
                "status_code": response.status_code,
                "error_from_tf_serving": response.text
            }, indent=2)

        # Parse TF Serving output
        data = response.json()
        outputs = data.get("outputs", {})

        # Extract values for the first batch item (index 0)
        formatted_response = {
            "verdict": {
                "classes": outputs.get("classes", [""])[0],
                "confidences": outputs.get("confidences", [0.0])[0]
            },
            "probabilities": {
                "roses": outputs.get("roses", [0.0])[0],
                "dandelion": outputs.get("dandelion", [0.0])[0],
                "tulips": outputs.get("tulips", [0.0])[0],
                "daisy": outputs.get("daisy", [0.0])[0],
                "sunflowers": outputs.get("sunflowers", [0.0])[0]
            }
        }

        # Return the formatted prediction
        return formatted_response["probabilities"]

    except requests.exceptions.ConnectionError:
        return json.dumps({"error": f"Failed to connect to TF Serving on {TENSORFLOW_SERVING_ENDPOINT}. Is the Docker container running?"}, indent=2)
    except requests.exceptions.Timeout:
        return json.dumps({"error": "Timeout while calling TF Serving."}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"}, indent=2)


def main():
    print("Starting Gradio interface...")

    with gr.Blocks(title="Flower Image Classifier") as demo:
        gr.Markdown("# Flower Image Classification 🌸")
        gr.Markdown("Upload a flower photo to get the prediction of its specie via TensorFlow Serving.")

        with gr.Row():
            with gr.Column():
                # type='pil' to avoid any client-side TensorFlow dependency
                input_image = gr.Image(label="Flower Image", type="pil")
                submit_btn = gr.Button("Classify", variant="primary")

            with gr.Column():
                output_dict = gr.Label(label="Prediction Result (TF Serving)")

        submit_btn.click(
            fn=predict,
            inputs=input_image,
            outputs=[output_dict]
        )

    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT
    )


if __name__ == "__main__":
    main()
