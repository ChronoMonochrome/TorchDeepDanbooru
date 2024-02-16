from PIL import Image
import numpy as np
import torch
import argparse
import os
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
    
def handle_img(img_filename, filemask, force):
    print(f"Image: {img_filename}")
    if filemask != "STDOUT":
        output_file = filemask % os.path.splitext(img_filename)[0]
        if os.path.exists(output_file) and not force:
            print(f"Output file {output_file} already exists. Skipping evaluation.")
        else:
            tags = evaluate(img_filename)
            with open(output_file, "w") as f:
                for tag, confidence in tags.items():
                    f.write(f"{tag}: {confidence}\n")
    else:
        tags = evaluate(img_filename)
        for tag, confidence in tags.items():
            print(f"{tag}: {confidence}")
            
def handle_dir(img_directory, filemask, force):
    for file in os.listdir(img_directory):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_filename = os.path.join(img_directory, file)
            print(f"Image: {img_filename}")
            if filemask != "STDOUT":
                output_file = filemask % os.path.splitext(img_filename)[0]
                if os.path.exists(output_file) and not force:
                    print(f"Output file {output_file} already exists. Skipping evaluation.")
                else:
                    tags = evaluate(img_filename)
                    with open(output_file, "w") as f:
                        for tag, confidence in tags.items():
                            f.write(f"{tag}: {confidence}\n")
            else:
                print(f"Image: {img_filename}")
                for tag, confidence in tags.items():
                    print(f"{tag}: {confidence}")
                    
def mandatory_flags(args):
    if not (args.image or args.directory):
        parser.error("Either -i/--image or -d/--directory should be provided.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Danbooru Image Tagging")
    parser.add_argument("-i", "--image", type=str, help="Path to image file")
    parser.add_argument("-d", "--directory", type=str, help="Path to directory containing images")
    parser.add_argument("-f", "--filemask", type=str, default="%s.txt", help="Output filename mask, use STDOUT to print tags instead")
    parser.add_argument("--force", action="store_true", help="Force re-evaluate and overwrite output file")

    args = parser.parse_args()
    mandatory_flags(args)

    g_cuda_available = torch.cuda.is_available()

    if g_cuda_available:
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    model = deep_danbooru_model.DeepDanbooruModel()
    model.load_state_dict(torch.load("model-resnet_custom_v3.pt"))

    model.eval()

    if g_cuda_available:
        model.half()
        model.cuda()

    if args.image:
        handle_img(args.image, args.filemask, args.force)
    elif args.directory:
        handle_dir(args.directory, args.filemask, args.force)