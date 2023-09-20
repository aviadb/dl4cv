# %%
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights,
    ResNet50_Weights, ResNet101_Weights,
    ResNet152_Weights
)
from torchvision.io import read_image
import pandas as pd

#%%

# img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

# model = torch.hub.load('pytorch/vision:v0.15.2', 'resnet101', pretrained=True)

# Step 1: Initialize model with the best available weights

models_list = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

model = dict.fromkeys(models_list)
weights = dict.fromkeys(models_list)
pre_proc = dict.fromkeys(models_list)
DS_ver = 'IMAGENET1K_V1'
# DS_ver = 'DEFAULT'
for net_model in models_list:
    weights[net_model] = eval('{}_Weights.{}'.format(net_model, DS_ver))
    ctor = net_model.lower()
    model[net_model] = eval(f'{ctor}(weights={weights[net_model]})')
    model[net_model].eval();
    pre_proc[net_model] = weights[net_model].transforms(antialias=True)

#%%
predict_df = pd.DataFrame(columns=['Model', 'img', 'Top-Class'])
                                   
                                   
                                #    , 'Prob.', 'Top3-Classes'])

# %%
img_path = 'imgs/dog.jpg'
img_path = 'imgs/ILSVRC2012_val_00035585.JPEG'
img_path = 'imgs/n02111889_7198.JPEG'
img = read_image(img_path)
T.ToPILImage()(img).show()
#%%
# prediction = []
for mdl_idx,mdl in enumerate(models_list):
    # batch[mdl] = 
    pred = model[mdl](pre_proc[mdl](img).unsqueeze(0)).squeeze(0).softmax(0)
    class_id = pred.argmax().item()
    pred_list = [
        mdl, 
        img_path,
        '{}: {:.2f}%'.format(
            weights[mdl].meta["categories"][class_id],
            100 * pred[class_id].item()
        )
        # f"{weights['ResNet50'].meta["categories"][class_id]}: {100 * pred[class_id].item():.2f}%"
    ]
    predict_df.loc[len(predict_df)] = pred_list
    # prediction.append(pred.)
#%%
predict_df
#%%
batch_res50 = preproc_rnet50(img).unsqueeze(0)
batch_res101 = preproc_rnet101(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model['ResNet50'](batch_res50).detach().squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights['ResNet50'].meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")



#%%





    # model[net_model] = net_model.lower()(weights=weights[net_model])
# weights = {
#     'resnet18': ResNet18_Weights.IMAGENET1K_V1,
#     'resnet34': ResNet34_Weights.IMAGENET1K_V1,
#     'resnet50': ResNet50_Weights.IMAGENET1K_V1,
#     'resnet101': ResNet101_Weights.IMAGENET1K_V1,
#     'resnet152': ResNet152_Weights.IMAGENET1K_V1
# }
# model['resnet18'] = resnet18(weights=weights['resnet18'])
# model['resnet34'] = resnet34(weights=weights['resnet34'])
# model['resnet50'] = resnet50(weights=weights['resnet50'])
# model['resnet101'] = resnet101(weights=weights['resnet101'])
# model['resnet152'] = resnet152(weights=weights['resnet152'])

# for net_model in model.keys():
#     model[net_model].eval();

# w_resnet18 = ResNet18_Weights.IMAGENET1K_V1
# rnet18 = resnet18(weights=w_resnet18)
# rnet18.eval();

# w_resnet34 = ResNet34_Weights.IMAGENET1K_V1
# rnet34 = resnet34(weights=w_resnet34)
# rnet34.eval();

# w_resnet50 = ResNet50_Weights.IMAGENET1K_V1
# rnet50 = resnet50(weights=w_resnet50)
# rnet50.eval();

# w_resnet101 = ResNet101_Weights.IMAGENET1K_V1
# rnet101 = resnet101(weights=w_resnet101)
# rnet101.eval();

# w_resnet152 = ResNet152_Weights.IMAGENET1K_V1
# rnet152 = resnet152(weights=w_resnet152)
# rnet152.eval();

# preproc_rnet18 = w_resnet18.transforms()
# preproc_rnet34 = w_resnet34.transforms()
# preproc_rnet50 = w_resnet50.transforms()
# preproc_rnet101 = w_resnet101.transforms()
# preproc_rnet152 = w_resnet152.transforms()

#%%
img = read_image("imgs/dog.jpg")
T.ToPILImage()(img).show()
#%%
# batch_res18 = preproc_rnet18(img).unsqueeze(0)
# batch_res34 = preproc_rnet34(img).unsqueeze(0)
batch_res50 = pre_proc['ResNet50'](img).unsqueeze(0)
batch_res101 = pre_proc['ResNet101'](img).unsqueeze(0)
# batch_res152 = preproc_rnet152(img).unsqueeze(0)

#%%
# predictions = pd.DataFrame()

#%%

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
batch_res50 = preproc_rnet50(img).unsqueeze(0)
batch_res101 = preproc_rnet101(img).unsqueeze(0)
#%%
# Step 4: Use the model and print the predicted category
prediction = model['ResNet50'](batch_res50).detach().squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights['ResNet50'].meta["categories"][class_id]
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
