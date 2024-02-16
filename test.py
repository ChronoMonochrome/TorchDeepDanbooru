from PIL import Image
import numpy as np
import torch
import tqdm

import deep_danbooru_model

def evaluate(img_filename):
    res = dict()
    pic = Image.open(img_filename).convert("RGB").resize((512, 512))
    a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

    if g_cuda_available:
        context = torch.no_grad(), torch.autocast("cuda")
    else:
        context = torch.no_grad()

    with context:
        x = torch.from_numpy(a)
        if g_cuda_available:
            x = x.cuda()

        # first run
        y = model(x)[0].detach().cpu().numpy()

        # measure performance
        for n in tqdm.tqdm(range(10)):
            model(x)


    for i, p in enumerate(y):
        if p >= 0.5:
            res[model.tags[i]] = p
            
    return res

if __name__ == "__main__":
    g_cuda_available = torch.cuda.is_available()

    if g_cuda_available:
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    model = deep_danbooru_model.DeepDanbooruModel()
    model.load_state_dict(torch.load('model-resnet_custom_v3.pt'))

    model.eval()

    if g_cuda_available:
        model.half()
        model.cuda()

    print(evaluate("test.jpg"))