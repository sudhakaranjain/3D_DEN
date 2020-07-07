from PIL import Image
import os
import json

folderpath = 'Flower_dataset'

with open('Flower_dataset/cat_to_name.json') as f:
  labels = json.load(f)


for subdir, dirs, files in os.walk(folderpath):
	break

for d in dirs:
	dir_path = os.path.join(folderpath, d)

	for file in os.listdir(dir_path):
		file_path = os.path.join(dir_path, file)
		# print(str(file_path))

		os.rename(str(file_path), str(dir_path)+"/"+labels[str(file)])

		# for directory in dirs:
		# 	labels_path = os.path.join(files_path, directory)
		# 	print(labels[str(label)])

				# if count2 == 1 or count2 == 6 or count2 == 10:
				# 	image_path = os.path.join(images_path, i)
				# 	im1 = Image.open(str(image_path))
				# 	rgb_im = im1.convert('RGB')
				# 	rgb_im = rgb_im.resize((100, 100), Image.ANTIALIAS)
				# 	new_path = "../Desktop/Flower_dataset_new/"+str(label)+"/"+str(files)+"/"+str(count1)+"/"+str(count2)+".jpg"
				# 	os.makedirs(os.path.dirname(new_path), exist_ok=True)
				# 	rgb_im.save(new_path)