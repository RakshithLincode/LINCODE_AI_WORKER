from mask_infrenece_module import *
import glob
import numpy as np
predictor = Inference()
for fname in glob.glob(r'D:\DATA\train\*.jpg'):
	frame = cv2.imread(fname)
	predictor.input_frame  = frame
	predicted_frame = predictor.dummy() 
	array = np.array(predicted_frame)
	c,img_w, img_h = array.shape
	unified_masks = numpy.zeros((img_w, img_h))
	for mask in array:
		unified_masks += mask
	unified_masks = unified_masks.astype(np.int32)
	plt.imsave('D:/Segmentatin_yolo/yolov5/segment/image/0.png', unified_masks,cmap='gray') 
