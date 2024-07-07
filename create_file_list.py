import os
import random
from glob import glob
from dataset import *
import torch
import torch.utils.data
from model import *
from transform import *
import re

def xfunc(_str):
	if(_str == '-1'):
		return '0'
	elif(_str == '1'):
		return '1'
	else:
		raise('class coding error')
	
def create_file_list(args):
	torch.manual_seed(args.seed) 
	random.seed(args.seed)

	# generate train_filelist.txt
	with open('../nfs3/datasets/celeba/Anno/list_attr_celeba.txt', 'r') as f0:
		attr_lines = f0.readlines()
		ti, si = -1, -1
		for i in range(40):
			if(args.true_attribute == attr_lines[1].split(' ')[i]):
				ti = i
				break
		for i in range(40):
			if(args.sensitive_attribute == attr_lines[1].split(' ')[i]):
				si = i
				break
		if (si == -1 or ti == -1):
			raise('The true attribute or the sensitive attribute is meaningless, please adjust.')
		

		with open("file_list/"+args.feature+"train_filelist.txt","w") as f1: # output
			all_dir = sorted(glob("../nfs3/datasets/celeba/train/*")) # Represents all labels of images
			split = len(args.varsigma) / args.nb_s_groups
			count1, count0 = 0, 0
			for i, dir in enumerate(all_dir): 
				all_image = sorted(glob(dir+"/images/*"))
				for i2, image in enumerate(all_image):
					num = int(os.path.basename(image).split('.')[0])
					sa = xfunc(re.split(r'[ \n]+',attr_lines[num+1])[si+1])
					if sa == '1':
						count1 += 1
					elif sa == '0':
						count0 += 1
					else:
						raise('sensitive attribute error.')
					break
			split1, split0 = count1//split + 1, count0//split + 1 # 47, 33 -> 24, 16
			cc1, cc0 = 0, 0
			for i, dir in enumerate(all_dir): 
				all_image = sorted(glob(dir+"/images/*"))
				for i2, image in enumerate(all_image):
					num = int(os.path.basename(image).split('.')[0])
					ta = xfunc(re.split(r'[ \n]+',attr_lines[num+1])[ti+1])
					sa = xfunc(re.split(r'[ \n]+',attr_lines[num+1])[si+1])
					if(sa == '1'):
						if i2 == 0:
							cc1 += 1
						f1.write(image + " " + ta + " " + sa + " " + str(int(cc1//split1 + split)) + "\n") 
					if(sa == '0'):
						if i2 == 0:
							cc0 += 1
						f1.write(image + " " + ta + " " + sa + " " + str(int(cc0//split0)) + "\n") 
				

		# generate test_filelist.txt
		with open("file_list/"+args.feature+"test_filelist.txt" , "w") as f1:
			all_dir = sorted(glob("../nfs3/datasets/celeba/test/*")) # Represents all labels of images
			for i, dir in enumerate(all_dir): # all_wnids
					all_image = sorted(glob(dir+"/*"))
					for i2, image in enumerate(all_image):
						num = int(os.path.basename(image).split('.')[0])
						ta = xfunc(re.split(r'[ \n]+',attr_lines[num+1])[ti+1])
						sa = xfunc(re.split(r'[ \n]+',attr_lines[num+1])[si+1])
						f1.write(image + " " + ta + " " + sa + "\n") 

	print("create_file_list finished")
	return