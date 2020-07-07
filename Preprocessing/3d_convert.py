from PIL import Image
import os

filepath = 'modelnet40'

for label in os.listdir(filepath):
	label_path = os.path.join(filepath, label)
	# if not os.path.exists('myfile.txt'):
	# 	break

	# list = len(os.listdir(label_path))
	for files in os.listdir(label_path):
		count1 = 0
		count2 = 0
		images_path = os.path.join(label_path, files)
		if str(files) == 'test' or str(files) == 'train':
			filenames = [img for img in os.listdir(images_path)]
			filenames.sort()
			for i in filenames:
				# print(str(img[-7:-3]))
				count2 = count2+1
				if count2 > 12:
					count2 = 1
					count1 = count1+1

				if count2 == 1 or count2 == 6 or count2 == 10:
					image_path = os.path.join(images_path, i)
					im1 = Image.open(str(image_path))
					rgb_im = im1.convert('RGB')
					rgb_im = rgb_im.resize((100, 100), Image.ANTIALIAS)
					new_path = "../Desktop/modelnet40_new/"+str(label)+"/"+str(files)+"/"+str(count1)+"/"+str(count2)+".jpg"
					os.makedirs(os.path.dirname(new_path), exist_ok=True)
					rgb_im.save(new_path)