import os
import json
import torch
import textwrap
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
import sys
sys.path.append("/workspace/dabin/YOLO-World/Qwen2.5-VL/qwen-vl-utils/src")
from qwen_vl_utils.vision_process import process_vision_info
# from qwen_vl_utils import process_vision_info  # Use if installed; otherwise, comment out
from tqdm import tqdm
import argparse
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import multiprocessing as mp
# 맨 위에 추가
GT_JSON_PATH = "/workspace/dabin/YOLO-World/data/russia/train/annotation/test_converted.json"

with open(GT_JSON_PATH, "r", encoding="utf-8-sig") as f:
    gt_data = json.load(f)
filename_to_imgid = {img["file_name"]: img["id"] for img in gt_data["images"]}

 
def wrap_text(text, font, max_width):
    """
    Wrap the given text using textwrap and compute the pixel width and height for each line.
    """
    wrapped_lines = []
    lines = textwrap.wrap(text, width=max_width)
    for line in lines:
        bbox = font.getbbox(line)
        w_line = bbox[2] - bbox[0]
        h_line = bbox[3] - bbox[1]
        wrapped_lines.append((line, w_line, h_line))
    return wrapped_lines
 
def letterbox_image(image, target_size=(1024, 1024)):
    """
    Resize 'image' while keeping its aspect ratio, then pad to target_size.
    Returns the padded image, scaling factor, and padding (pad_x, pad_y).
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
    try:
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
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        print("[DEBUG] Inputs prepared.")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        return output_text

    except Exception as e:
        print("[ERROR] Inference failed:", e)
        return "[]"

def plot_bounding_boxes_on_original(orig_img, boxes_json, scale, pad_x, pad_y, target_size=(1024, 1024)):
    """
    Parse the model's JSON output (in letterboxed coordinates) and draw bounding boxes on the original image.
    Robust to malformed or incomplete output.
    """
    boxes_json = boxes_json.strip()
    if boxes_json.startswith("```json"):
        boxes_json = boxes_json[len("```json"):].strip()
    if boxes_json.endswith("```"):
        boxes_json = boxes_json[:-3].strip()

    try:
        boxes = json.loads(boxes_json)
        if not isinstance(boxes, list):
            print("[WARNING] Model output is not a list. Skipping this image.")
            return orig_img
    except Exception as e:
        print("[ERROR] JSON parsing error:", e)
        return orig_img

    draw = ImageDraw.Draw(orig_img)
    font = ImageFont.load_default()

    for box in boxes:
        bbox = box.get("bbox_2d", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            print(f"[WARNING] Skipping invalid bbox: {bbox}")
            continue

        try:
            x1, y1, x2, y2 = bbox
            # Convert letterboxed coordinates to original image coordinates
            orig_x1 = int((x1 - pad_x) / scale)
            orig_y1 = int((y1 - pad_y) / scale)
            orig_x2 = int((x2 - pad_x) / scale)
            orig_y2 = int((y2 - pad_y) / scale)

            # 정렬 (x1 ≤ x2, y1 ≤ y2)
            x_min = min(orig_x1, orig_x2)
            x_max = max(orig_x1, orig_x2)
            y_min = min(orig_y1, orig_y2)
            y_max = max(orig_y1, orig_y2)

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            label = box.get("label", "unknown")
            draw.text((x_min + 4, y_min + 4), label, fill="red", font=font)

        except Exception as e:
            print(f"[ERROR] Failed to draw box {bbox}: {e}")
            continue

    return orig_img

 
def evaluate_image_files(image_files, output_dir, model, processor):
    """
    Process a list of image files:
      1. Record original resolution.
      2. Letterbox-resize the image to 1024x1024 while preserving aspect ratio.
      3. Run inference with a prompt emphasizing distinctive class features.
      4. Convert predicted boxes (in letterboxed coordinates) back to original coordinates.
      5. Save images with bounding boxes and generate COCO-format annotations.
    """
    coco_images = []
    coco_annotations = []
    category_mapping = {}
    annotation_id = 0
    # image_id = 0
    # Soft
    # new_class_descriptions = (
    #     "tank: A tracked military vehicle with a visible turret or gun barrel and armored body.\n"
    #     "armored car: A wheeled vehicle with an armored exterior or mounted weapon system.\n"
    #     "military truck: A logistics truck with military adaptations such as camouflage or reinforced chassis.\n"
    #     "Do not detect antennas, partial objects, shadows, terrain, roads, people, or buildings. "
    #     "Each vehicle should be marked with a single bounding box only."
    # )
    
    # # Medium
    # new_class_descriptions = (
    #     "tank: A mostly visible tracked vehicle with features similar to T-64, T-72, or T-80. "
    #     "Must include a turret or long gun barrel and tank-like armor plating.\n"
    #     "armored car: A wheeled armored vehicle resembling BMD-2’s turret shape, BMP series’ low-profile hull, "
    #     "or BTR-70/80’s cylindrical armored body. May include small-caliber weapons.\n"
    #     "military truck: A truck such as BM-21 with visible military adaptations—camouflage paint, reinforced chassis, "
    #     "or mounted rocket pods.\n"
    #     "Do not detect antennas, partial objects, shadows, terrain, roads, people, or buildings. "
    #     "Each vehicle should be marked with a single bounding box only."
    # )
    # Hard
    new_class_descriptions = (
        "tank: A fully detailed tracked combat vehicle of T-64/T-72/T-80 class. "
        "T-64 features composite armor and an autoloading 125 mm gun; T-72 adds welded rolled steel armor and NBC protection; "
        "T-80 offers a gas-turbine engine and optional ERA modules. Must show turret, barrel, and track assembly.\n"
        "armored car: An 8×8 or 4×4 wheeled vehicle with armored plating and weapon mount. "
        "BMD-2: airborne IFV with 30 mm autocannon and low silhouette; BMP-1: 73 mm SPG-9 launcher; BMP-2: 30 mm two-stage cannon; "
        "BTR-70/80: dual-engine configuration with turreted MGs; MT-LB: multi-role tracked APC with flat deck and light armor.\n"
        "military truck: A BM-21 “Grad” launcher truck with 122 mm rocket pod racks, reinforced frame, and military camo. "
        "Chassis must show rocket rails or ammo storage racks.\n"
        "Do not detect antennas, partial objects, shadows, terrain, roads, people, or buildings. "
        "Each vehicle should be marked with a single bounding box only."
    )
    # Soft
    # prompt = ("You are analyzing a drone or ground-view image to detect any object that appears to be a military vehicle.\n"
    #     "Predict vehicles that resemble one of the following classes, even if the object is partially visible or not clearly identifiable.\n"
    #     "Military-looking features such as camouflage, shape, or attachments are enough for prediction. Minor ambiguity is acceptable.\n"
    #     "Do not include background, people, buildings, or shadows. Avoid detecting clearly unrelated objects.\n"
    #     "If no such vehicle appears, return an empty list: [].\n\n"
    #     "Class list and descriptions:\n"
    #     f"{new_class_descriptions}\n\n"
    #     "Output the bounding box coordinates and the corresponding label in JSON format as a list of objects.\n"
    #     "Each object must be a dictionary with keys 'bbox_2d' (as [x1, y1, x2, y2]) and 'label'.\n"
    #     "Do not include any additional text or explanation. Output only valid JSON."
    # )
    
    # Medium
    # prompt = ("You are analyzing a drone or ground-view image to detect visible military vehicles.\n"
        # "Predict vehicles that match one of the following classes if they show reasonable military features such as armor, camouflage, or mounted gear.\n"
        # "Vehicles should be mostly visible, but minor occlusions are acceptable. Avoid detecting ambiguous or civilian-looking vehicles.\n"
        # "Exclude partial parts like antennas or wheels, and do not include people, buildings, or shadows.\n"
        # "If no vehicle is likely to match, return an empty list: [].\n\n"
        # "Class list and descriptions:\n"
        # f"{new_class_descriptions}\n\n"
        # "Output the bounding box coordinates and the corresponding label in JSON format as a list of objects.\n"
        # "Each object must be a dictionary with keys 'bbox_2d' (as [x1, y1, x2, y2]) and 'label'.\n"
        # "Do not include any additional text or explanation. Output only valid JSON."
        # )
        
    # Hard
    prompt = (
        "You are analyzing a drone or ground-view image to detect fully visible military vehicles.\n"
        "Only predict vehicles that clearly match one of the following classes based on distinct military features such as turret, armor plating, or camouflage.\n"
        "Exclude vehicles that are partially visible, ambiguous, or lack obvious military characteristics.\n"
        "Do not include antennas, wheels, turrets, or other isolated parts. Also exclude background, shadows, people, or buildings.\n"
        "If no vehicle clearly fits the description, return an empty list: [].\n\n"
        "From the following class list and descriptions, select the class that best matches the object's distinctive characteristics:\n\n"
        "Class list and descriptions:\n"
        f"{new_class_descriptions}\n\n"
        "Output the bounding box coordinates and the corresponding label in JSON format as a list of objects.\n"
        "Each object must be a dictionary with keys 'bbox_2d' (as [x1, y1, x2, y2]) and 'label'.\n"
        "Do not include any additional text or explanation. Output only valid JSON."
    )

    orig_imgs = []
    orig_resolutions = []
    filenames = []
    output_texts = []
    letterbox_params = []  # To store (scale, pad_x, pad_y) for each image
 
    for image_file in tqdm(image_files, desc="Processing images on current GPU"):
        orig_img = Image.open(image_file).convert("RGB")
        orig_res = orig_img.size
        # Letterbox resize to preserve aspect ratio
        letterbox_img, scale, pad_x, pad_y = letterbox_image(orig_img, target_size=(1024,1024))
        output_text = inference(letterbox_img, prompt, model, processor)
        print(f"Model output for {os.path.basename(image_file)}:")
        print(output_text)
        output_texts.append(output_text)
        orig_imgs.append(orig_img)
        orig_resolutions.append(orig_res)
        filenames.append(os.path.basename(image_file))
        letterbox_params.append((scale, pad_x, pad_y))
 
        # Save image with drawn bounding boxes (convert predicted coordinates back)
        img_with_boxes = plot_bounding_boxes_on_original(orig_img.copy(), output_text, scale, pad_x, pad_y, target_size=(1024,1024))
        out_image_path = os.path.join(output_dir, os.path.basename(image_file))
        img_with_boxes.save(out_image_path)
        print(f"Saved output image to {out_image_path}")
 
    for idx, output_text in enumerate(output_texts):
        orig_width, orig_height = orig_resolutions[idx]
        file_name = filenames[idx]
        image_id = filename_to_imgid[file_name]
        coco_images.append({
            "id": image_id,
            "file_name": filenames[idx],
            "width": orig_width,
            "height": orig_height
        })
 
        try:
            boxes = json.loads(output_text.strip().strip("```json").strip("```"))
        except Exception as e:
            print(f"JSON parsing error for {filenames[idx]}: {e}")
            boxes = []
 
        for box in boxes:
            x1, y1, x2, y2 = box["bbox_2d"]
            # Convert letterboxed coordinates back to original image coordinates
            scale, pad_x, pad_y = letterbox_params[idx]
            abs_x1 = int((x1 - pad_x) / scale)
            abs_y1 = int((y1 - pad_y) / scale)
            abs_x2 = int((x2 - pad_x) / scale)
            abs_y2 = int((y2 - pad_y) / scale)
            width_box = abs_x2 - abs_x1
            height_box = abs_y2 - abs_y1
            area = width_box * height_box
            label = box.get("label", "")
            if label.lower() not in category_mapping:
                category_mapping[label.lower()] = len(category_mapping) + 1
            cat_id = category_mapping[label.lower()]
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [abs_x1, abs_y1, width_box, height_box],
                "area": area,
                "iscrowd": 0,
                "score": 1.0,
                "justification": box.get("justification", "")
            })
            annotation_id += 1
 
        image_id += 1
 
    coco_categories = [{"id": cid, "name": name} for name, cid in category_mapping.items()]
    coco_output = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }
    json_path = os.path.join(output_dir, "detection_results_coco.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
    print(f"Partition COCO JSON saved to {json_path}")
    return json_path
 
def process_images_on_gpu(gpu_id, image_files, output_dir, model_id):
    device = f"cuda:{gpu_id}"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"GPU {gpu_id} processing {len(image_files)} images.")
   
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    _ = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
   
    output_subdir = os.path.join(output_dir, f"gpu_{gpu_id}")
    os.makedirs(output_subdir, exist_ok=True)
   
    json_path = evaluate_image_files(image_files, output_subdir, model, processor)
    return json_path
 
def merge_json_results(output_dir, num_gpus):
    combined_images = []
    combined_annotations = []
    combined_categories = {}
   
    global_img_id = 1
    global_ann_id = 1
   
    for gpu_id in range(num_gpus):
        subdir = os.path.join(output_dir, f"gpu_{gpu_id}")
        json_file = os.path.join(subdir, "detection_results_coco.json")
        if not os.path.exists(json_file):
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
       
        local_img_map = {}
        for img in data["images"]:
            new_img = img.copy()
            local_id = new_img["id"]
            new_img["id"] = global_img_id
            local_img_map[local_id] = global_img_id
            combined_images.append(new_img)
            global_img_id += 1
       
        local_cat_map = {cat["id"]: cat["name"] for cat in data["categories"]}
       
        for ann in data["annotations"]:
            new_ann = ann.copy()
            new_ann["image_id"] = local_img_map.get(ann["image_id"], ann["image_id"])
            local_cat_name = local_cat_map.get(ann["category_id"], "")
            if local_cat_name not in combined_categories:
                combined_categories[local_cat_name] = len(combined_categories) + 1
            new_ann["category_id"] = combined_categories[local_cat_name]
            new_ann["id"] = global_ann_id
            combined_annotations.append(new_ann)
            global_ann_id += 1
   
    categories_list = [{"id": cid, "name": name} for name, cid in combined_categories.items()]
    merged = {
        "images": combined_images,
        "annotations": combined_annotations,
        "categories": categories_list
    }
   
    merged_path = os.path.join(output_dir, "detection_results_coco_merged.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)
    print(f"Merged COCO JSON saved to {merged_path}")
    return merged_path
 
def main():
    parser = argparse.ArgumentParser(description="Multi-GPU zero-shot military object localization and annotation with aspect ratio preservation")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images and JSON")
    args = parser.parse_args()
 
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
   
    valid_exts = [".jpg", ".jpeg", ".png"]
    all_images = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                  if os.path.splitext(f)[1].lower() in valid_exts]
    if not all_images:
        print("No images found in the input directory.")
        return
 
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
   
    partitions = [[] for _ in range(num_gpus)]
    for idx, img_path in enumerate(all_images):
        partitions[idx % num_gpus].append(img_path)
   
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=process_images_on_gpu, args=(gpu_id, partitions[gpu_id], args.output_dir, model_id))
        p.start()
        processes.append(p)
   
    for p in processes:
        p.join()
    print("Processing completed on all GPUs.")
   
    merge_json_results(args.output_dir, num_gpus)
 
if __name__ == "__main__":
    main()



