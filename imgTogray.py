import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import argparse

def rgb2gray(rgb):

	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.299 * r + 0.587 * g + 0.114 * b
	return gray

def show_img(img_path):
	img = mpimg.imread(img_path)
	gray = rgb2gray(img)
	plt.imshow(gray, cmap = plt.get_cmap('gray'))
	plt.show()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', default='', help='img to change.')
	args = parser.parse_args()

	if args.path != '':
		path = args.path
		show_img(path)
	else:
		print('please enter path of image.')

if __name__ == '__main__':
	main()