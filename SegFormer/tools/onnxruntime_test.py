import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

idname = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', \
          'traffic sign', 'vegetation', 'terrain' , 'sky' , 'person' , 'rider'   , 'car' , 'truck' , 'bus', 'train',  \
          'motorcycle', 'bicycle']

cityspallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]


def get_iou(pred, target, nclass):
    ious = []
    for i in range(nclass):
        pred_ins = pred == i 
        target_ins = target == i
        inser = pred_ins[target_ins].sum()
        union = pred_ins.sum() + target_ins.sum() - inser
        if union != 0:
            iou = inser / union
        else:
            iou = 0
        ious.append(iou)
    
    return ious


def get_color_pallete(npimg, dataset='city'):

    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(cityspallete)
    return out_img


def label_id_map(target):
    # id map(34 -> 19)
    label = np.ones(target.shape, dtype=np.int32) * 255
    label[target == 7] = 0
    label[target == 8] = 1
    label[target == 11] = 2
    label[target == 12] = 3
    label[target == 13] = 4
    label[target == 17] = 5
    label[target == 19] = 6
    label[target == 20] = 7
    label[target == 21] = 8
    label[target == 22] = 9
    label[target == 23] = 10
    label[target == 24] = 11
    label[target == 25] = 12
    label[target == 26] = 13
    label[target == 27] = 14
    label[target == 28] = 15
    label[target == 31] = 17
    label[target == 32] = 17
    label[target == 33] = 18
    return label
 
 
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


img_path = "/root/data/leftImg8bit/val/frankfurt/frankfurt_000000_000576_leftImg8bit.png"
img = imread(img_path)

providers = [
	('CUDAExecutionProvider', {
		'device_id': 0,
	})
]

onnx_model = onnx.load_model("/root/onnx/segformer.b2.1024x1024.city.160k.onnx")
sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=providers)

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

output = sess.run([output_name], {input_name : img})
output = np.squeeze(output[0])


target = Image.open("/root/data/gtFine/val/frankfurt/frankfurt_000000_000576_gtFine_labelIds.png")
target = target.resize((1024, 1024))
target = np.array(target)
target = label_id_map(target)

unique, counts = np.unique(output, return_counts=True)
print(dict(zip(unique, counts)))

unique, counts = np.unique(target, return_counts=True)
print(dict(zip(unique, counts)))

iou_list = get_iou(output, target, 19)

for i, n in zip(iou_list, idname):
    print(n, i)

mask = get_color_pallete(output, 'citys')
mask.save("out.png")