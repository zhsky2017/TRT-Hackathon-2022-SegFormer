import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from glob import glob

def np_normalize(data, mean, std):
    # transforms.ToTensor, transforms.Normalize的numpy 实现
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean)
    if not isinstance(std, np.ndarray):
        std = np.array(std)
    if mean.ndim == 1:
        mean = np.reshape(mean, (-1, 1, 1))
    if std.ndim == 1:
        std = np.reshape(std, (-1, 1, 1))
    _max = np.max(abs(data))
    _div = np.divide(data, _max)  # i.e. _div = data / _max
    _sub = np.subtract(_div, mean)  # i.e. arrays = _div - mean
    arrays = np.divide(_sub, std)  # i.e. arrays = (_div - mean) / std
    arrays = np.transpose(arrays, (2, 0, 1))
    return arrays

def imread(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB').resize((1024, 1024))
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3))
    std  = np.array([0.229, 0.224, 0.225]).reshape((1, 3))
    img = np_normalize(img, mean, std)
    img = np.expand_dims(img, 0)
    return img.astype(np.float32)

batch_size_list = [1, 2, 4, 8]

img_path_list = glob("/root/data/leftImg8bit/val/frankfurt/*_leftImg8bit.png")

onnx_model = onnx.load_model("/root/onnx/segformer.b2.1024x1024.city.160k.onnx")
providers = [
	('CUDAExecutionProvider', {
		'device_id': 0,
	})
]
sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=providers)

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

for b in batch_size_list:
    concat_list = []
    iodata = {}
    for p in img_path_list[: b]:
        img = imread(p)
        concat_list.append(img)

    img = np.concatenate(concat_list, axis=0)
    # iodata['input'] = img

    output = sess.run([output_name], {input_name : img})
    output = np.squeeze(output[0])
    # iodata['output'] = output

    np.savez("segformer-b{}".format(b), input=img, output=output)









