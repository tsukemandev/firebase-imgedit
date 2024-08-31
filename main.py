# The Cloud Functions for Firebase SDK to create Cloud Functions and set up triggers.
from firebase_functions import firestore_fn, https_fn, options

# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app



import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import io

import tensorflow as tf



app = initialize_app()



class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.compat.v1.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]



def vis_segmentation2(image, seg_map):

  # 기존 코드의 수정된 부분
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  # 첫 번째 subplot: 다운로드한 이미지를 표시
  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')



def vis_segmentation(image, seg_map):


  # 이미지 크기 가져오기
  img_width, img_height = image.size

  # 이미지 크기에 맞게 plt.figure 사이즈 설정 (인치 단위로 변환)
  # DPI는 보통 100으로 가정하지만, 이미지 DPI에 따라 다를 수 있습니다.
  dpi = 100.0
  figsize = (img_width / dpi, img_height / dpi)

  # figure 생성
  plt.figure(figsize=figsize)
  

  # 여백 없애기
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

  plt.imshow(image)
  plt.axis('off')




LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def generate_plot(req: https_fn.Request) -> https_fn.Response:
    # Create a simple plot
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1, 2, 3], [0, 1, 4, 9], marker='o')
    plt.title("Sample Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100.0)
    buf.seek(0)

    return https_fn.Response(buf.getvalue(), headers={'Content-Type': 'image/png'})



@https_fn.on_request(memory=options.MemoryOption.GB_1, timeout_sec=540, cors=options.CorsOptions(
        cors_origins=[r"firebase\.com$", r"https://flutter\.com", r"http://localhost:3000", 
                      r"http://imgdit\.com", r"https://imgdit\.com", r"imgdit\.com$", 
                        r"app-imgdit\.web\.app", r"app-imgdit\.firebaseapp\.com", 
                        r"http://app-imgdit\.web\.app", r"https://app-imgdit\.web\.app", 
                        r"http://app-imgdit\.firebaseapp\.com", r"https://app-imgdit\.firebaseapp\.com",
                        r"www\.imgdit\.com", r"http://www\.imgdit\.com", r"https://www\.imgdit\.com"],
        cors_methods=["get", "post"],
    ))
def convert_image(req: https_fn.Request) -> https_fn.Response:


  # POST 요청에서 이미지 데이터 가져오기
  if req.method != 'POST':
      return https_fn.Response('Only POST requests are allowed.', status=405)
  
  # 파일 가져오기 (multipart/form-data 방식)
  image_file = req.files.get('image')  # 'image'는 클라이언트에서 전송된 파일 필드명
  if image_file is None:
      return https_fn.Response('No image file provided.', status=400)



  model_dir = os.path.dirname(os.path.abspath(__file__))
  model_path = os.path.join(model_dir, 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz')

  MODEL = DeepLabModel(model_path)
  print('model loaded successfully!')

  
  # IMAGE_URL = 'https://picsum.photos/id/237/200/300'  #@param {type:"string"}

  resized_im=''
  seg_map=''


  try:
    #f = urllib.request.urlopen(IMAGE_URL)
    #jpeg_str = f.read()
    #original_im = Image.open(BytesIO(jpeg_str))

    original_im = Image.open(image_file )
  except IOError:
    return


  resized_im, seg_map = MODEL.run(original_im)

  # Apply white background to the image based on segmentation
  white_bg_image = apply_white_background(resized_im, seg_map)

  # Visualize the results
  vis_segmentation(white_bg_image, seg_map)

  # vis_segmentation(resized_im, seg_map)



  # Save plot to a bytes buffer
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # buf.seek(0)

  return https_fn.Response(buf.getvalue(), headers={'Content-Type': 'image/png'})
  # return https_fn.Response(f"Message with ID added.")


 
def apply_white_background(image, seg_map):
  """Applies a white background to the areas of the image segmented as background.

  Args:
    image: The original resized image (PIL.Image).
    seg_map: The segmentation map from the DeepLab model.

  Returns:
    Image with the background set to white.
  """
  # Convert PIL image to numpy array
  image_array = np.array(image)

  # Create a white background
  white_background = np.ones_like(image_array) * 255

  # Set background (label 0) to white
  mask = seg_map == 0  # assuming '0' is the label for background
  image_array[mask] = white_background[mask]

  # Convert back to PIL image
  white_bg_image = Image.fromarray(image_array)

  return white_bg_image

def run_visualization_with_white_bg(url, MODEL):
  """Inferences DeepLab model, applies white background, and visualizes result."""
  try:
      f = urllib.request.urlopen(url)
      jpeg_str = f.read()
      original_im = Image.open(BytesIO(jpeg_str))
  except IOError:
      print('Cannot retrieve image. Please check url: ' + url)
      return

  print('Running deeplab on image %s...' % url)
  resized_im, seg_map = MODEL.run(original_im)

  # Apply white background to the image based on segmentation
  white_bg_image = apply_white_background(resized_im, seg_map)

  # Visualize the results
  vis_segmentation(white_bg_image, seg_map)