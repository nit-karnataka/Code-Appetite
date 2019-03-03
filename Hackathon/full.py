import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import math
import math
import cv2
import numpy as np


def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):
	"""Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
	
	Args:
		img: grayscale uint8 image of the text-line to be segmented.
		kernelSize: size of filter kernel, must be an odd integer.
		sigma: standard deviation of Gaussian function used for filter kernel.
		theta: approximated width/height ratio of words, filter function is distorted by this factor.
		minArea: ignore word candidates smaller than specified area.
		
	Returns:
		List of tuples. Each tuple contains the bounding box and the image of the segmented word.
	"""

	# apply filter kernel
	kernel = createKernel(kernelSize, sigma, theta)
	imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
	(_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	imgThres = 255 - imgThres

	# find connected components. OpenCV: return type differs between OpenCV2 and 3
	if cv2.__version__.startswith('3.'):
		(_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		(components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# append components to result
	res = []
	for c in components:
		# skip small word candidates
		if cv2.contourArea(c) < minArea:
			continue
		# append bounding box and image of word to result list
		currBox = cv2.boundingRect(c) # returns (x, y, w, h)
		(x, y, w, h) = currBox
		currImg = img[y:y+h, x:x+w]
		res.append((currBox, currImg))

	# return list of words, sorted by x-coordinate
	return sorted(res, key=lambda entry:entry[0][0])


def prepareImg(img, height):
	"""convert given image to grayscale image (if needed) and resize to desired height"""
	assert img.ndim in (2, 3)
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	h = img.shape[0]
	factor = height / h
	return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
	"""create anisotropic filter kernel according to given parameters"""
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2
	
	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta
	
	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize
			
			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
			
			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel

def main():
	"""reads images from data/ and outputs the word-segmentation to out/"""

	# read input images from 'in' directory
	imgFiles = os.listdir('../sentence/')
	for (i,f) in enumerate(imgFiles):
		print('Segmenting words of sample %s'%f)
		
		# read image, prepare it by resizing it to fixed height and converting it to grayscale
		img = prepareImg(cv2.imread('../sentence/%s'%f), 50)
		
		# execute segmentation with given parameters
		# -kernelSize: size of filter kernel (odd integer)
		# -sigma: standard deviation of Gaussian function used for filter kernel
		# -theta: approximated width/height ratio of words, filter function is distorted by this factor
		# - minArea: ignore word candidates smaller than specified area
		res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=200)
		
		# write output to 'out/inputFileName' directory
		if not os.path.exists('../out/%s'%f):
			os.mkdir('../out/%s'%f)
		
		# iterate over all segmented words
		print('Segmented into %d words'%len(res))
		for (j, w) in enumerate(res):
			(wordBox, wordImg) = w
			(x, y, w, h) = wordBox
			cv2.imwrite('../image/%s.png'%(f), wordImg) # save word
			cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
		
		# output summary image with bounding boxes around words
		#cv2.imwrite('../out/%s/summary.png'%f, img)

pdf_dir = r"C:\\Users\\win\\Desktop\\now\\pdf"
os.chdir(pdf_dir)
img_dir = r"C:\\Users\\win\\Desktop\\now\\images1"
for pdf_file in os.listdir(pdf_dir):
	if pdf_file.endswith(".pdf"):
		pages = convert_from_path(pdf_file,300)
		pdf_file = pdf_file[:-4]
		for page in pages:
			os.chdir(img_dir)
			page.save("%s-page%d.jpg" % (pdf_file,pages.index(page)), "JPEG")
			os.chdir(pdf_dir)
		os.remove(pdf_file+".pdf")
crp_dir = r"C:\\Users\\win\\Desktop\\now\\crop"
data_dir = r"C:\\Users\\win\\Desktop\\now\\sentence"


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images    
images=load_images_from_folder("C:\\Users\\win\\Desktop\\now\\images")
for image in images:
	edges = cv2.Canny(image, threshold1=50, threshold2=255)
	im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	try: hierarchy = hierarchy[0]
	except: hierarchy = []
	height, width = edges.shape
	min_x, min_y = width, height
	max_x = max_y = 0
	for contour, hier in zip(contours, hierarchy):
		(x,y,w,h) = cv2.boundingRect(contour)
		min_x, max_x = min(x, min_x), max(x+w, max_x)
		min_y, max_y = min(y, min_y), max(y+h, max_y)
		if w > 80 and h > 80:
			cv2.rectangle(edges, (x,y), (x+w,y+h), (255, 0, 0), 2)
	if max_x - min_x > 0 and max_y - min_y > 0:
		cv2.rectangle(edges, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
	Crp_img = image[min_y:max_y,min_x:max_x]
	p = Crp_img.copy()
	os.chdir(crp_dir)
	cv2.imwrite("0.png",Crp_img)
	os.chdir(data_dir)
	n = 0
	for i in range(Crp_img.shape[0]):
		j = Crp_img[n:n+100,:]
		n = n+100
		if n>= Crp_img.shape[0]:
			break
		cv2.imwrite("%d.png"%i ,j)
	os.chdir(img_dir)
if __name__ == '__main__':
	main()

