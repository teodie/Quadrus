from os import listdir

from PIL import Image
from rembg import new_session, remove

input_path = "../sample_images"
output_path = "../output_images"

model = "u2net_human_seg"

sample_images = listdir(input_path)

for image in sample_images:
  input = Image.open(f"{input_path}/{image}")
  session = new_session(model)
  output = remove(input, session=session, bgcolor=(255, 255, 255, 255))
  output.save(f"{output_path}/{image.split(".")[0] + ".png"}")
