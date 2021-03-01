from PIL import Image
import os

filepath = 'RGB-D'

# for train_or_test in os.listdir(filepath):
# 	projections_path = os.path.join(filepath, train_or_test) + '/RGB'

# 	for projection in os.listdir(projections_path):
# 		labels_path = os.path.join(projections_path, projection)
# 		for label in os.listdir(labels_path):

# 			# print(label)
# 			images_path = os.path.join(labels_path, label)
# 			filenames = [int(img[:-4]) for img in os.listdir(images_path)]
# 			filenames.sort()
# 			# print(filenames)

# 			count = 0
# 			for i in filenames:
				
# 				img_path = os.path.join(images_path, str(i)) + ".jpg"
# 				im1 = Image.open(str(img_path))
# 				if im1.getbbox():
# 					rgb_im = im1.convert('RGB')
# 					rgb_im = rgb_im.resize((128, 128), Image.ANTIALIAS)
# 					new_path = "./RGB-D_new/"+str(train_or_test)+"/"+str(projection)+"/"+str(label)+"/"+str(count)+".jpg"
# 					os.makedirs(os.path.dirname(new_path), exist_ok=True)
# 					rgb_im.save(new_path)
# 					count = count + 1


for train_or_test in os.listdir(filepath): 
	projections_path = os.path.join(filepath, train_or_test) + '/RGB'
	projections = os.listdir(projections_path)
	
	labels_path0 = os.path.join(projections_path, projections[0])
	labels_path1 = os.path.join(projections_path, projections[1])
	labels_path2 = os.path.join(projections_path, projections[2])

	labels =  os.listdir(labels_path0)

	for label in labels:

		# print(label)
		images_path0 = os.path.join(labels_path0, label)
		images_path1 = os.path.join(labels_path1, label)
		images_path2 = os.path.join(labels_path2, label)

		filenames0 = [int(img[:-4]) for img in os.listdir(images_path0)]
		filenames0.sort()
		filenames1 = [int(img[:-4]) for img in os.listdir(images_path1)]
		filenames1.sort()
		filenames2 = [int(img[:-4]) for img in os.listdir(images_path2)]
		filenames2.sort()

		count = 0
		for idx, i in enumerate(filenames0):
			
			img_path0 = os.path.join(images_path0, str(i)) + ".jpg"
			img_path1 = os.path.join(images_path1, str(filenames1[idx])) + ".jpg"
			img_path2 = os.path.join(images_path2, str(filenames2[idx])) + ".jpg"


			im0 = Image.open(str(img_path0))
			im1 = Image.open(str(img_path1))
			im2 = Image.open(str(img_path2))

			if im0.getbbox() and im1.getbbox() and im2.getbbox():

				rgb_im = im0.convert('RGB')
				rgb_im = rgb_im.resize((128, 128), Image.ANTIALIAS)
				new_path = "./RGB-D_new/"+str(train_or_test)+"/"+str(projections[0])+"/"+str(label)+"/"+str(count)+".jpg"
				os.makedirs(os.path.dirname(new_path), exist_ok=True)
				rgb_im.save(new_path)

				rgb_im = im1.convert('RGB')
				rgb_im = rgb_im.resize((128, 128), Image.ANTIALIAS)
				new_path = "./RGB-D_new/"+str(train_or_test)+"/"+str(projections[1])+"/"+str(label)+"/"+str(count)+".jpg"
				os.makedirs(os.path.dirname(new_path), exist_ok=True)
				rgb_im.save(new_path)

				rgb_im = im2.convert('RGB')
				rgb_im = rgb_im.resize((128, 128), Image.ANTIALIAS)
				new_path = "./RGB-D_new/"+str(train_or_test)+"/"+str(projections[2])+"/"+str(label)+"/"+str(count)+".jpg"
				os.makedirs(os.path.dirname(new_path), exist_ok=True)
				rgb_im.save(new_path)

				count = count + 1

