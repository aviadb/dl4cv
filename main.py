# %%
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
from torchvision.io import read_image

#%%

# img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")
img = read_image("imgs/dog.jpg")

# model = torch.hub.load('pytorch/vision:v0.15.2', 'resnet101', pretrained=True)

# Step 1: Initialize model with the best available weights
w_resnet50 = ResNet50_Weights.IMAGENET1K_V1
rnet50 = resnet50(weights=w_resnet50)
rnet50.eval();

w_resnet101 = ResNet101_Weights.IMAGENET1K_V1
rnet101 = resnet101(weights=w_resnet101)
rnet101.eval();

preproc_rnet50 = w_resnet50.transforms()
preproc_rnet101 = w_resnet101.transforms()

# preprocess = weights.transforms()
# input_tensor = preprocess(img)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

# with torch.no_grad():
#     output = model(input_batch)
# # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# # print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# # print(probabilities)

# class_id = probabilities.argmax().item()
# score = probabilities[class_id].item()
# category_name = weights.meta["categories"][class_id]

# print(f"{category_name}: {100 * score:.1f}%")

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
[top5_scores, top5_idx] = prediction.topk(5)
cat_list = [ weights.meta["categories"][idx] for idx in top5_idx ]

#%%
for idx,res in  enumerate(cat_list):
    print(f"{res}: {100 * top5_scores[idx]:.1f}%")

#%%
T.ToPILImage()(img)

#%%

img = read_image("imgs/n02111889_7198.JPEG") # Example image from training set
T.ToPILImage()(img).show()

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

img = read_image("imgs/ILSVRC2012_val_00035585.JPEG") # Another example from training set
T.ToPILImage()(img).show()

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
