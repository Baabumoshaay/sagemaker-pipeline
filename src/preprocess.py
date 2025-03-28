import argparse
import os
from PIL import Image

def preprocess_images(input_dir, output_dir, image_size=(128, 128)):
    sets = ['train', 'test']
    classes = ['REAL', 'FAKE']

    for dataset in sets:
        for cls in classes:
            input_path = os.path.join(input_dir, dataset, cls)
            output_path = os.path.join(output_dir, dataset, cls)
            os.makedirs(output_path, exist_ok=True)

            for filename in os.listdir(input_path):
                try:
                    img = Image.open(os.path.join(input_path, filename)).convert("RGB")
                    img = img.resize(image_size)
                    img.save(os.path.join(output_path, filename))
                except Exception as e:
                    print(f"Skipping {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    preprocess_images(args.input_dir, args.output_dir)