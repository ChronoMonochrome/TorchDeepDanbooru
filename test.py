from PIL import Image
import numpy as np
import torch
import tqdm

import deep_danbooru_model

cuda_available = torch.cuda.is_available()

if cuda_available:
    device = torch.device('cuda')
    print("CUDA is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load('model-resnet_custom_v3.pt'))

model.eval()

if cuda_available:
    model.half()
    model.cuda()

pic = Image.open("test.jpg").convert("RGB").resize((512, 512))
a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

if cuda_available:
    context = torch.no_grad(), torch.autocast("cuda")
else:
    context = torch.no_grad()

with context:
    x = torch.from_numpy(a)
    if cuda_available:
        x = x.cuda()

    # first run
    y = model(x)[0].detach().cpu().numpy()

    # measure performance
    for n in tqdm.tqdm(range(10)):
        model(x)


for i, p in enumerate(y):
    if p >= 0.5:
        print(model.tags[i], p)
