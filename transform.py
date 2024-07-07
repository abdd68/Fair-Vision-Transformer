import torchvision.transforms as transforms
'''transform的性质'''

transform_image = transforms.Compose([transforms.Resize((224, 224)),
									  transforms.ToTensor(),
									  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
	                                                 std = [ 1/0.229, 1/0.224, 1/0.225 ]),
	                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
	                                                 std = [ 1., 1., 1. ]),])