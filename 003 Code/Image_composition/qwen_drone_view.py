import os
import json
import torch
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer

def letterbox_image(image, target_size=(640, 640)):
    """
    Resize the image while preserving aspect ratio, then pad with gray color to target_size.
    Returns the padded image, scaling factor, and x/y padding.
    """
    target_w, target_h = target_size
    orig_w, orig_h = image.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    new_image = Image.new("RGB", target_size, (128, 128, 128))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    new_image.paste(resized, (pad_x, pad_y))
    return new_image, scale, pad_x, pad_y

def inference(image, prompt, model, processor):
    """
    Run the Qwen model inference using the given image and text prompt.
    Returns the model's generated text (trimmed).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Try to process vision info if available (if qwen_vl_utils is installed)
    image_inputs, video_inputs = None, None
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    except ImportError:
        image_inputs = image

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=2,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        early_stopping=True
    )
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return output_text.strip()

# ---------------------------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Drone-View Classification Script using Qwen")
    parser.add_argument("--image_dir", type=str, default='/mnt/d/py/aim/projects/drone_detection/ovd/dataset_datamaker/train_dataset/OD_2/images', help="Directory containing the input images")
    parser.add_argument("--output_dir", type=str, default='/mnt/d/py/aim/projects/drone_detection/ovd/dataset_datamaker/train_dataset/fusion/train/', help="Directory to save classification results")
    args = parser.parse_args()
    
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    _ = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    
    # List image files with extensions jpg, jpeg, png
    image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                   if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png"]]
    if not image_files:
        print("No image files found in the directory.")
        return

    classification_results = {}
    # Qwen prompt for image classification
    prompt = (
        "Based on this image, decide whether it is taken from a drone view or a normal view. "
        "Answer only with 'Drone View' or 'Normal View' without any additional explanation."
    )
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            orig_img = Image.open(image_file).convert("RGB")
        except Exception as e:
            print(f"Failed to open image ({image_file}): {e}")
            continue
        letterbox_img, scale, pad_x, pad_y = letterbox_image(orig_img, target_size=(640, 640))
        result = inference(letterbox_img, prompt, model, processor)
        # Determine classification based on the result
        is_drone_view = True if "Drone View" in result else False
        classification_results[os.path.basename(image_file)] = "Drone View" if is_drone_view else "Normal View"
        print(f"{os.path.basename(image_file)}: {classification_results[os.path.basename(image_file)]}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_json_path = os.path.join(args.output_dir, "drone_view_classification.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(classification_results, f, ensure_ascii=False, indent=4)
    print(f"Classification results saved to {output_json_path}")

if __name__ == "__main__":
    main()
