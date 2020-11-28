

import os
from glob import glob
from zipfile import ZipFile

import numpy as np
import skimage

from ._tqdm import tqdm



def highres_image(path, img_as_float=True):

	path = path if path[-1] in {'/', '\\'} else (path + '/')
	hr = skimage.io.imread(path + 'HR.png', dtype=np.uint16)
	sm = skimage.io.imread(path + 'SM.png', dtype=np.bool)
	if img_as_float:
		hr = skimage.img_as_float64(hr)
	return (hr, sm)
	


def lowres_image_iterator(path, img_as_float=True):


	path = path if path[-1] in {'/', '\\'} else (path + '/')
	for f in glob(path + 'LR*.png'):
		q = f.replace('LR', 'QM')
		l = skimage.io.imread(f, dtype=np.uint16)
		c = skimage.io.imread(q, dtype=np.bool)
		if img_as_float:
			l = skimage.img_as_float64(l)
		yield (l, c)
	


def check_img_as_float(img, validate=True):

	if not issubclass(img.dtype.type, np.floating):
		img = skimage.img_as_float64(img)
	# https://scikit-image.org/docs/dev/api/skimage.html#img-as-float64
	
	if validate:
		# safeguard against unwanted conversions to values outside the
		# [0.0, 1.0] range (would happen if `img` had signed values).
		assert img.min() >= 0.0 and img.max() <= 1.0
	
	return img
	




def all_scenes_paths(base_path):

	base_path = base_path if base_path[-1] in {'/', '\\'} else (base_path + '/')
	return [
		base_path + c + s
		for c in ['RED/', 'NIR/']
		for s in sorted(os.listdir(base_path + c))
		]
	


def scene_id(scene_path, incl_channel=False):

	sep = os.path.normpath(scene_path).split(os.sep)
	if incl_channel:
		return '/'.join(sep[-2:])
	else:
		return sep[-1]
	



def prepare_submission(images, scenes, subm_fname='submission.zip'):
	
	assert len(images) == 290, '%d images provided, 290 expected.' % len(images)
	assert len(images) == len(scenes), "Mismatch in number of images and scenes."
	assert subm_fname[-4:] == '.zip'
	
	# specific warnings we wish to ignore
	warns = [
		'tmp.png is a low contrast image',
		'Possible precision loss when converting from float64 to uint16']
	
	with np.warnings.catch_warnings():
		for w in warns:
			np.warnings.filterwarnings('ignore', w)
		
		print('Preparing submission. Writing to "%s".' % subm_fname)
		
		with ZipFile(subm_fname, mode='w') as zf:
			
			for img, scene in zip(tqdm(images), scenes):
				assert img.shape == (384, 384), \
					'Wrong dimensions in image for scene %s.' % scene
				
				skimage.io.imsave('tmp.png', img)
				zf.write('tmp.png', arcname=scene_id(scene) + '.png')
		
		os.remove('tmp.png')
	

