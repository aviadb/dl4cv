#%%
using Metalhead
using Images
using Flux
using Flux: onecold
using DataAugmentation

model = ResNet(101, pretrain=true);

#%%
img = Images.load("imgs/ILSVRC2012_val_00035585.JPEG")
img_data = apply(augmentations, Image(img)) |> itemdata

# img = Images.load("imgs/dog.jpg")
# img_data = apply(augmentations, Image(img)) |> itemdata

#%%

predictions = model(Flux.unsqueeze(img_data, 4))
predict = softmax(predictions)
println(maximum(predict))
print(onecold(predictions, labels))
#%%

DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)

augmentations = CenterCrop((224, 224)) |>
                ImageToTensor() |>
                Normalize(DATA_MEAN, DATA_STD)

data = apply(augmentations, Image(img)) |> itemdata

# ImageNet labels
labels = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))

println(onecold(model(Flux.unsqueeze(data, 4)), labels))