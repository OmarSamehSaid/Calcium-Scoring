import tensorflow as tf
import numpy as np
import pydicom
from typing import Optional
from scipy.ndimage import measurements
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to non-interactive mode
import matplotlib.pyplot as plt
app = Flask(__name__)
CORS(app)
loaded_model = tf.saved_model.load("./N")
def get_object_agatston(calc_object: np.ndarray, calc_pixel_count: int):
  object_max = np.max(calc_object)
  object_agatston = 0
  if 130 <= object_max < 200:
    object_agatston = calc_pixel_count * 1
  elif 200 <= object_max < 300:
    object_agatston = calc_pixel_count * 2
  elif 300 <= object_max < 400:
    object_agatston = calc_pixel_count * 3
  elif object_max >= 400:
    object_agatston = calc_pixel_count * 4
  return object_agatston
def compute_agatston_for_slice(ds, predicted_mask: Optional[np.ndarray],
                               min_calc_object_pixels = 3) -> int:
  def create_hu_image(ds):
    return ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
  if not predicted_mask is None:
    mask = predicted_mask
  else:
    mask = example.ground_truth_mask
  if np.sum(mask) == 0:
    return 0
  slice_agatston = 0
  pixel_volume = (ds.PixelSpacing[0] * ds.PixelSpacing[1])
  example=ds               
  hu_image = create_hu_image(example)
  labeled_mask, num_labels = measurements.label(mask,
                                                structure=np.ones((3, 3)))
  for calc_idx in range(1, num_labels + 1):
    label = np.zeros(mask.shape)
    label[labeled_mask == calc_idx] = 1
    calc_object = hu_image * label

    calc_pixel_count = np.sum(label)
    if calc_pixel_count <= min_calc_object_pixels:
      continue
    calc_volume = calc_pixel_count * pixel_volume
    object_agatston = round(get_object_agatston(calc_object, calc_volume))
    slice_agatston += object_agatston
  return slice_agatston
def prepare_input_image(image, expand_dims):
    image = np.expand_dims(image, axis=0)
    if expand_dims:
        return np.expand_dims(image, axis=3)
    else:
        return np.expand_dims(image, axis=2)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list')
def list_files():
    return render_template('list.html')

@app.route('/calculate_agatston', methods=['POST'])
def calculate_agatston_score():
    try:
        file = request.files['dicom_file']
        ds = pydicom.dcmread(file)
        image_array = (ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept).astype(np.float32)
        predictions = loaded_model(prepare_input_image(image_array, True), training=False).numpy()
        predictions = np.squeeze(predictions)
        binarized_prediction = (predictions > 0.1).astype(np.float32)
        agatston_score = compute_agatston_for_slice(ds, binarized_prediction)

        plt.imshow(image_array, cmap='gray')
        plt.imshow(predictions, cmap='viridis', alpha=0.6)
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        seg_data_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.clf()
        plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        image_data_base64 = base64.b64encode(buffer.getvalue()).decode()


        return jsonify({"agatston_score": agatston_score, "image_data": image_data_base64,"seg_data":seg_data_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/calculate_agatstons', methods=['POST'])
def calculate_agatston_scores():
    try:
        uploaded_files = request.files.getlist('dicom_files')
        agatston_scores = []
        seg_images = []

        for file in uploaded_files:
            ds = pydicom.dcmread(file)
            image_array = (ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept).astype(np.float32)
            predictions = loaded_model(prepare_input_image(image_array, True), training=False).numpy()
            predictions = np.squeeze(predictions)
            binarized_prediction = (predictions > 0.1).astype(np.float32)
            agatston_score = compute_agatston_for_slice(ds, binarized_prediction)

            if agatston_score > 0:
                plt.imshow(image_array, cmap='gray')
                plt.imshow(predictions, cmap='viridis', alpha=0.6)
                plt.axis('off')
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                seg_data_base64 = base64.b64encode(buffer.getvalue()).decode()
                seg_images.append(seg_data_base64)

            agatston_scores.append(agatston_score)

        total_agatston_score = sum(agatston_scores)

        return jsonify({
            "total_agatston_score": total_agatston_score,
            "seg_images": seg_images
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
