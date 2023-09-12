# %%
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image

#%%

# img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")
img = read_image("imgs/dog.jpg")


# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.eval();

#%%

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).detach().squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

# %%
[top5_scores, top5_idx] = prediction.topk(5)
cat_list = [ weights.meta["categories"][idx] for idx in top5_idx ]

#%%
for idx,res in  enumerate(cat_list):
    print(f"{res}: {100 * top5_scores[idx]:.1f}%")

#%%
T.ToPILImage()(img)

#%%

img = read_image("imgs/n02111889_7198.JPEG")
T.ToPILImage()(img)

#%%
# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).detach().squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

#%%

img = read_image("imgs/ILSVRC2012_val_00035585.JPEG")
T.ToPILImage()(img)

#%%
# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).detach().squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")


# %%
# !wget 'https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8NHx8fGVufDB8fHx8&w=1000&q=80'  -O white_horse.jpg
# %%
img = read_image("imgs/white_horse.jpg")
T.ToPILImage()(img)
# %%
# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).detach().squeeze(0).softmax(0)
[top5_scores, top5_idx] = prediction.topk(5)
cat_list = [ weights.meta["categories"][idx] for idx in top5_idx ]

for idx,res in  enumerate(cat_list):
    print(f"{res}: {100 * top5_scores[idx]:.1f}%")

# %%
img_rotated = T.functional.rotate(img, 180)
T.ToPILImage()(img_rotated)
# %%
batch = preprocess(img_rotated).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).detach().squeeze(0).softmax(0)
[top5_scores, top5_idx] = prediction.topk(5)
cat_list = [ weights.meta["categories"][idx] for idx in top5_idx ]

for idx,res in  enumerate(cat_list):
    print(f"{res}: {100 * top5_scores[idx]:.1f}%")

# %%
img_flipped = T.functional.hflip(img)
T.ToPILImage()(img_flipped)
# %%
batch = preprocess(img_flipped).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).detach().squeeze(0).softmax(0)
[top5_scores, top5_idx] = prediction.topk(5)
cat_list = [ weights.meta["categories"][idx] for idx in top5_idx ]

for idx,res in  enumerate(cat_list):
    print(f"{res}: {100 * top5_scores[idx]:.1f}%")

# %%
img_inverted = T.functional.invert(img)
T.ToPILImage()(img_inverted)
# %%
batch = preprocess(img_inverted).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).detach().squeeze(0).softmax(0)
[top5_scores, top5_idx] = prediction.topk(5)
cat_list = [ weights.meta["categories"][idx] for idx in top5_idx ]

for idx,res in  enumerate(cat_list):
    print(f"{res}: {100 * top5_scores[idx]:.1f}%")
