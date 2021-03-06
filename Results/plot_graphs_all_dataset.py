import numpy as np
import random
import cv2 as cv
import gzip
import matplotlib.pyplot as plt

fig, a = plt.subplots(2, 3, figsize=(40, 20))


# ------------------------- ModelNet40 --------------------------

acc_mobile = np.load("MobileNetV2/all_accuracies.npy")
acc_vgg = np.load("VGG16/all_accuracies.npy")
acc_custom = np.load("Custom/all_accuracies.npy")

time_mobile = np.load("MobileNetV2/all_time.npy")
time_vgg = np.load("VGG16/all_time.npy")
time_custom = np.load("Custom/all_time.npy")

size_mobile = np.load("MobileNetV2/s_all.npy")
size_vgg = np.load("VGG16/s_all.npy")
size_custom = np.load("Custom/s_all.npy")

# -------- Accuracies -----------

vgg = [0] * len(acc_vgg[0])
for acc in acc_vgg:
	for i in range(len(acc)):
		vgg[i] = vgg[i] + acc[i]

vgg = np.array(vgg) / 10.0

apa_vgg = np.sum(vgg) / 40
print("Avg. Per-Task Accuracy for VGG: ",apa_vgg)

mobile = [0] * len(acc_mobile[0])
for acc in acc_mobile:
	for i in range(len(acc)):
		mobile[i] = mobile[i] + acc[i]

mobile = np.array(mobile) / 10.0

apa_mobile = np.sum(mobile) / 40
print("Avg. Per-Task Accuracy for MobileNetV2: ",apa_mobile)

custom = [0] * len(acc_custom[0])
for acc in acc_custom:
	for i in range(len(acc)):
		custom[i] = custom[i] + acc[i]

custom = np.array(custom) / 10.0

apa_custom = np.sum(custom) / 40
print("Avg. Per-Task Accuracy for Custom Model: ",apa_custom)

# print(s)

tasks = [i+1 for i in range(len(acc_vgg[0]))]
trials = [i+1 for i in range(10)]
# print(tasks)

# axes= a[0].add_axes([0.1,0.1,0.8,0.4])

a[0,0].plot(tasks, mobile, marker='s', linestyle='dashed', ms=5)
a[0,0].plot(tasks, vgg, marker='*', linestyle='dashed')
a[0,0].plot(tasks, custom, marker='D', linestyle='dashed', ms=5)
a[0,0].set_xlim(0,40)
a[0,0].set_ylim(0.50,1.05)
a[0,0].legend(["MobileNetV2", "VGG16", "Custom"])

a[0,0].minorticks_on()

# plt.xticks(np.arange(0, max(tasks)+1, 5))
# a[0].set_yticks(np.arange(0.50, 1.05, 0.05))
# a[0].tick_params(labelsize='large')
a[0,0].grid(b=True, which='major', color='#666666', linestyle='-')
a[0,0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

a[0,0].set_xlabel("Tasks / Number of Categories")
a[0,0].set_ylabel("Global Classification Accuracy")
# plt.xticks(tasks, tasks, rotation='horizontal')

# a[0,0].annotate('ModelNet40', xy=(0, 0.5), xytext=(-a[0,0].yaxis.labelpad - 5, 0), rotation=90,
#                 xycoords=a[0,0].yaxis.label, textcoords='offset points',
#                 size='large', ha='right', va='center')


# -------- Training Time -----------

t_mobile = [0] * len(time_mobile[0])
for acc in time_mobile:
	for i in range(len(acc)):
		t_mobile[i] = t_mobile[i] + acc[i]

t_mobile = np.array(t_mobile) / 10.0


t_vgg = [0] * len(time_vgg[0])
for acc in time_vgg:
	for i in range(len(acc)):
		t_vgg[i] = t_vgg[i] + acc[i]

t_vgg = np.array(t_vgg) / 10.0


t_custom = [0] * len(time_custom[0])
for acc in time_custom:
	for i in range(len(acc)):
		t_custom[i] = t_custom[i] + acc[i]

t_custom = np.array(t_custom) / 10.0

a[0,1].plot(tasks, t_mobile, marker='s', linestyle='dashed', ms=5)
a[0,1].plot(tasks, t_vgg, marker='*', linestyle='dashed')
a[0,1].plot(tasks, t_custom, marker='D', linestyle='dashed', ms=5)
a[0,1].set_xlim(0,40)
# a[1].set_ylim(0.0, 110, 10)
# a[0].legend(["MobileNetV2", "VGG16", "Custom"])

a[0,1].minorticks_on()

# a[1].set_xticks(np.arange(0, max(tasks)+1, 5))
# a[1].set_yticks(np.arange(0, 120, 10))
# a[1].tick_params(labelsize='large')
a[0,1].grid(b=True, which='major', color='#666666', linestyle='-')
a[0,1].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

a[0,1].set_xlabel("Tasks / Number of Categories")
a[0,1].set_ylabel("Training Time (in sec)")

a[0,1].annotate('ModelNet40', xy=(0.3, 1), xytext=(0, 310),
                xycoords=a[0,1].xaxis.label, textcoords='offset points',
                size='large', ha='left', va='center')


# -------- Layer Sizes -----------

s_mobile = [0] * len(size_mobile[0])
for s in size_mobile:
	for i in range(len(s)):
		l = s[i]
		s_mobile[i] = s_mobile[i] + l[0] + l[1]

s_mobile = np.array(s_mobile) / 10.0

s_vgg = [0] * len(size_vgg[0])
for s in size_vgg:
	for i in range(len(s)):
		l = s[i]
		s_vgg[i] = s_vgg[i] + l[0] + l[1]

s_vgg = np.array(s_vgg) / 10.0

s_custom = [0] * len(size_custom[0])
for s in size_custom:
	for i in range(len(s)):
		l = s[i]
		s_custom[i] = s_custom[i] + l[0] + l[1]

s_custom = np.array(s_custom) / 10.0

a[0,2].plot(tasks, s_mobile, marker='s', linestyle='dashed', ms=5)
a[0,2].plot(tasks, s_vgg, marker='*', linestyle='dashed')
a[0,2].plot(tasks, s_custom, marker='D', linestyle='dashed', ms=5)
a[0,2].set_xlim(0,40)
a[0,2].set_ylim(1100, 1800)
# a[0].legend(["MobileNetV2", "VGG16", "Custom"])

a[0,2].minorticks_on()

# a[2].set_xticks(np.arange(0, max(tasks)+1, 1))
# a[2].set_yticks(np.arange(0, 120, 10))
# a[2].tick_params(labelsize='large')
a[0,2].grid(b=True, which='major', color='#666666', linestyle='-')
a[0,2].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

a[0,2].set_xlabel("Tasks / Number of Categories")
a[0,2].set_ylabel("Total Neurons in DEN layers")



# ------------------------- RGB-D --------------------------

acc_mobile = np.load("RGB-D/MobileNetV2/all_accuracies.npy")
acc_vgg = np.load("RGB-D/VGG16/all_accuracies.npy")
acc_custom = np.load("RGB-D/Custom/all_accuracies.npy")

time_mobile = np.load("RGB-D/MobileNetV2/all_time.npy")
time_vgg = np.load("RGB-D/VGG16/all_time.npy")
time_custom = np.load("RGB-D/Custom/all_time.npy")

size_mobile = np.load("RGB-D/MobileNetV2/s_all.npy")
size_vgg = np.load("RGB-D/VGG16/s_all.npy")
size_custom = np.load("RGB-D/Custom/s_all.npy")

# -------- Accuracies -----------

vgg = [0] * len(acc_vgg[0])
for acc in acc_vgg:
	for i in range(len(acc)):
		vgg[i] = vgg[i] + acc[i]

vgg = np.array(vgg) / 10.0

apa_vgg = np.sum(vgg) / 51
print("Avg. Per-Task Accuracy for VGG: ",apa_vgg)

print("\n GCA VGG16: ",vgg[-1])

mobile = [0] * len(acc_mobile[0])
for acc in acc_mobile:
	for i in range(len(acc)):
		mobile[i] = mobile[i] + acc[i]

mobile = np.array(mobile) / 10.0

apa_mobile = np.sum(mobile) / 51
print("Avg. Per-Task Accuracy for MobileNetV2: ",apa_mobile)

print("\n GCA MobileNetV2: ",mobile[-1])

custom = [0] * len(acc_custom[0])
for acc in acc_custom:
	for i in range(len(acc)):
		custom[i] = custom[i] + acc[i]

custom = np.array(custom) / 10.0

apa_custom = np.sum(custom) / 51
print("Avg. Per-Task Accuracy for Custom Model: ",apa_custom)

print("\n GCA Custom: ",custom[-1])

# print(s)

tasks = [i+1 for i in range(len(acc_vgg[0]))]
trials = [i+1 for i in range(10)]
# print(tasks)

# axes= a[0].add_axes([0.1,0.1,0.8,0.4])

a[1,0].plot(tasks, mobile, marker='s', linestyle='dashed', ms=5)
a[1,0].plot(tasks, vgg, marker='*', linestyle='dashed')
a[1,0].plot(tasks, custom, marker='D', linestyle='dashed', ms=5)
a[1,0].set_xlim(0,51)
a[1,0].set_ylim(0.20,1.05)
# a[1,0].legend(["MobileNetV2", "VGG16", "Custom"])

a[1,0].minorticks_on()

# plt.xticks(np.arange(0, max(tasks)+1, 5))
# a[0].set_yticks(np.arange(0.50, 1.05, 0.05))
# a[0].tick_params(labelsize='large')
a[1,0].grid(b=True, which='major', color='#666666', linestyle='-')
a[1,0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

a[1,0].set_xlabel("Tasks / Number of Categories")
a[1,0].set_ylabel("Global Classification Accuracy")
# plt.xticks(tasks, tasks, rotation='horizontal')

# -------- Training Time -----------

t_mobile = [0] * len(time_mobile[0])
for acc in time_mobile:
	for i in range(len(acc)):
		t_mobile[i] = t_mobile[i] + acc[i]

t_mobile = np.array(t_mobile) / 10.0


t_vgg = [0] * len(time_vgg[0])
for acc in time_vgg:
	for i in range(len(acc)):
		t_vgg[i] = t_vgg[i] + acc[i]

t_vgg = np.array(t_vgg) / 10.0


t_custom = [0] * len(time_custom[0])
for acc in time_custom:
	for i in range(len(acc)):
		t_custom[i] = t_custom[i] + acc[i]

t_custom = np.array(t_custom) / 10.0

a[1,1].plot(tasks, t_mobile, marker='s', linestyle='dashed', ms=5)
a[1,1].plot(tasks, t_vgg, marker='*', linestyle='dashed')
a[1,1].plot(tasks, t_custom, marker='D', linestyle='dashed', ms=5)
a[1,1].set_xlim(0,51)
# a[1].set_ylim(0.0, 110, 10)
# a[0].legend(["MobileNetV2", "VGG16", "Custom"])

a[1,1].minorticks_on()

# a[1].set_xticks(np.arange(0, max(tasks)+1, 5))
# a[1].set_yticks(np.arange(0, 120, 10))
# a[1].tick_params(labelsize='large')
a[1,1].grid(b=True, which='major', color='#666666', linestyle='-')
a[1,1].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

a[1,1].set_xlabel("Tasks / Number of Categories")
a[1,1].set_ylabel("Training Time (in sec)")

a[1,1].annotate('RGB-D', xy=(0.4, 1), xytext=(0, 310),
                xycoords=a[1,1].xaxis.label, textcoords='offset points',
                size='large', ha='left', va='center')



# -------- Layer Sizes -----------

s_mobile = [0] * len(size_mobile[0])
for s in size_mobile:
	for i in range(len(s)):
		l = s[i]
		s_mobile[i] = s_mobile[i] + l[0] + l[1]

s_mobile = np.array(s_mobile) / 10.0

s_vgg = [0] * len(size_vgg[0])
for s in size_vgg:
	for i in range(len(s)):
		l = s[i]
		s_vgg[i] = s_vgg[i] + l[0] + l[1]

s_vgg = np.array(s_vgg) / 10.0

s_custom = [0] * len(size_custom[0])
for s in size_custom:
	for i in range(len(s)):
		l = s[i]
		s_custom[i] = s_custom[i] + l[0] + l[1]

s_custom = np.array(s_custom) / 10.0

a[1,2].plot(tasks, s_mobile, marker='s', linestyle='dashed', ms=5)
a[1,2].plot(tasks, s_vgg, marker='*', linestyle='dashed')
a[1,2].plot(tasks, s_custom, marker='D', linestyle='dashed', ms=5)
a[1,2].set_xlim(0,51)
a[1,2].set_ylim(1100, 1800)
# a[0].legend(["MobileNetV2", "VGG16", "Custom"])

a[1,2].minorticks_on()

# a[2].set_xticks(np.arange(0, max(tasks)+1, 1))
# a[2].set_yticks(np.arange(0, 120, 10))
# a[2].tick_params(labelsize='large')
a[1,2].grid(b=True, which='major', color='#666666', linestyle='-')
a[1,2].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

a[1,2].set_xlabel("Tasks / Number of Categories")
a[1,2].set_ylabel("Total Neurons in DEN layers")

# plt.axis([0.0, 40, 0.7, 1])

# fig.tight_layout()
plt.show()