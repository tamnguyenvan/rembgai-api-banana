import io
from typing import Tuple
import base64
import torch
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort


def init():
    # Init is ran on server startup
    # Load your model to GPU as a global variable here using the variable name "model"

    global model

    model_path = 'isnet-general-use.onnx'
    if torch.cuda.is_available():
        model = ort.InferenceSession(model_path, providers=[
                                     'CUDAExecutionProvider'])
    else:
        model = ort.InferenceSession(model_path, providers=[
                                     'CPUExecutionProvider'])


def preprocess(image_b64: str, input_size: int) -> Tuple:
    nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
    im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h0, w0 = im0.shape[:2]
    image = cv2.resize(im0, (input_size, input_size))
    image = image[:, :, ::-1]
    image = image.astype(np.float32) / 255.
    image = (image - 0.5) / 1.
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    return im0, image, (h0, w0)


def postprocess(
    im0: np.ndarray,
    output: np.ndarray,
    h0: int,
    w0: int,
    return_mask: bool = False
) -> str:
    output = np.squeeze(output, (0, 1))
    output = cv2.resize(output, (w0, h0), cv2.INTER_LANCZOS4)
    mask = np.clip(output * 255, 0, 255).astype(np.uint8)
    if return_mask:
        # Encode
        output_str = cv2.imencode('.png', mask)[1].tostring()
        output_b64 = base64.b64encode(output_str).decode('utf-8')
    else:
        # Encode
        im0 = Image.fromarray(im0[:, :, ::-1])
        mask = Image.fromarray(mask)
        empty = Image.new("RGBA", (im0.size), 0)
        composite = Image.composite(im0, empty, mask)
        out = io.BytesIO()
        composite.save(out, 'png')
        out.seek(0)
        output_b64 = base64.b64encode(out.getvalue()).decode('utf-8')
    return output_b64


def inference(model_inputs: dict) -> dict:
    # Inference is ran for every server call
    # Reference your preloaded global model variable here.
    global model

    # Parse out your arguments
    image_b64 = model_inputs.get('image_b64', None)
    if image_b64 == None:
        return {'message': 'No image found'}

    # Run the model
    input_details = model.get_inputs()
    input_size = input_details[0].shape[2]
    im0, inputs, (h0, w0) = preprocess(image_b64, input_size)
    result = model.run(None, {input_details[0].name: inputs})[0]
    return_mask = model_inputs.get('return_mask')
    output_b64 = postprocess(im0, result, h0, w0, return_mask=return_mask)

    # Return the results as a dictionary
    return {'result': output_b64}
