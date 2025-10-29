# import os
# import json
# import torch
# import re
# import numpy as np
# from PIL import Image, ImageDraw
# from glob import glob
# from tqdm import tqdm
# import random
# import argparse
# import concurrent.futures
# import shutil  # 이미지 복사를 위해 추가

# from transformers import (
#     AutoProcessor, 
#     AutoModelForCausalLM, 
#     AutoTokenizer, 
#     T5Tokenizer, 
#     T5ForConditionalGeneration
# )

# # ----------------------------------------
# # 1. 모델 및 토크나이저 로드 함수
# # ----------------------------------------
# def load_models(device, torch_dtype):
#     fl_model = AutoModelForCausalLM.from_pretrained(
#         "microsoft/Florence-2-large",
#         torch_dtype=torch_dtype,
#         trust_remote_code=True
#     ).to(device)
#     fl_processor = AutoProcessor.from_pretrained(
#         "microsoft/Florence-2-large",
#         trust_remote_code=True
#     )
#     t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
#     t5_model = T5ForConditionalGeneration.from_pretrained(
#         "google/flan-t5-large", 
#         torch_dtype=torch_dtype
#     ).to(device)
#     t5_model.eval()
#     return fl_model, fl_processor, t5_tokenizer, t5_model

# def run_t5_yesno_query(query, t5_tokenizer, t5_model, device):
#     raw_inputs = t5_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
#     inputs = {k: v.to(device) for k, v in raw_inputs.items()}
#     outputs = t5_model.generate(**inputs, max_new_tokens=16)
#     answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
#     return answer

# # ----------------------------------------
# # 2. 문자 오프셋 및 bbox 관련 함수
# # ----------------------------------------
# def get_word_offsets(text):
#     offsets = []
#     for match in re.finditer(r'\S+', text):
#         offsets.append([match.start(), match.end()])
#     return offsets

# def ensure_offset_pair(t):
#     """
#     주어진 t가 단일 int이면 [t, t+1]로, 
#     길이가 1인 리스트/튜플이면 [t[0], t[0]+1]로, 
#     길이가 2인 경우 그대로 반환합니다.
#     그 외에는 None을 반환합니다.
#     """
#     if isinstance(t, int):
#         return [t, t+1]
#     elif isinstance(t, (list, tuple)):
#         if len(t) == 1:
#             return [t[0], t[0] + 1]
#         elif len(t) == 2:
#             return t
#     return None

# def safe_parse_tokens(caption, tokens_positive):
#     """
#     주어진 caption과 tokens_positive의 각 오프셋 정보를
#     ensure_offset_pair를 통해 [start, end] 형태로 보정하고,
#     해당 부분 문자열을 추출하여 리스트로 반환합니다.
#     인덱스 범위를 초과하면 해당 항목은 무시합니다.
#     """
#     parsed_tokens = []
#     for t in tokens_positive:
#         pair = ensure_offset_pair(t)
#         if pair is None:
#             continue
#         start, end = pair
#         if start < 0 or end > len(caption):
#             continue
#         parsed_tokens.append(caption[start:end])
#     return parsed_tokens

# def convert_bbox(bbox):
#     x1, y1, x2, y2 = bbox
#     return [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)]

# def combine_images_horizontally(img1, img2):
#     # 두 이미지의 높이를 동일하게 맞춘 후 좌우 결합
#     h = max(img1.height, img2.height)
#     w = img1.width + img2.width
#     combined = Image.new("RGB", (w, h))
#     combined.paste(img1, (0, 0))
#     combined.paste(img2, (img1.width, 0))
#     return combined

# def draw_bboxes(image, bboxes, color="red", width=3):
#     draw = ImageDraw.Draw(image)
#     for bbox in bboxes:
#         draw.rectangle(bbox, outline=color, width=width)
#     return image

# # ----------------------------------------
# # 3. 단일 이미지 처리 함수
# # ----------------------------------------
# def process_image(image_path, global_image_id, fl_model, fl_processor, t5_tokenizer, t5_model, device, 
#                   target_class="tank", dataset_name="flickr", category_id=1):
#     try:
#         image = Image.open(image_path).convert("RGB")
#     except Exception as e:
#         return None, None, None, None, None

#     # 캡션 생성 (프롬프트: "<DETAILED_CAPTION>")
#     prompt_caption = "<MORE_DETAILED_CAPTION>"
#     inputs_caption = fl_processor(text=prompt_caption, images=image, return_tensors="pt").to(device)
#     generated_ids_caption = fl_model.generate(
#         input_ids=inputs_caption["input_ids"],
#         pixel_values=inputs_caption["pixel_values"],
#         max_new_tokens=1024,
#         num_beams=3,
#         do_sample=False
#     )
#     generated_text_caption = fl_processor.batch_decode(generated_ids_caption, skip_special_tokens=False)[0]
#     caption_dict = fl_processor.post_process_generation(
#         generated_text_caption,
#         task=prompt_caption,
#         image_size=(image.width, image.height)
#     )
#     caption_text = caption_dict.get(prompt_caption, "")

#     # Phrase Grounding (프롬프트: "<CAPTION_TO_PHRASE_GROUNDING>")
#     prompt_grounding = "<CAPTION_TO_PHRASE_GROUNDING>"
#     grounding_input_text = prompt_grounding + " " + caption_text
#     inputs_grounding = fl_processor(text=grounding_input_text, images=image, return_tensors="pt").to(device)
#     generated_ids_grounding = fl_model.generate(
#         input_ids=inputs_grounding["input_ids"],
#         pixel_values=inputs_grounding["pixel_values"],
#         max_new_tokens=1024,
#         num_beams=3,
#         do_sample=False
#     )
#     generated_text_grounding = fl_processor.batch_decode(generated_ids_grounding, skip_special_tokens=False)[0]
#     grounding_result = fl_processor.post_process_generation(
#         generated_text_grounding,
#         task=prompt_grounding,
#         image_size=(image.width, image.height)
#     )

#     grounding_data = grounding_result.get(prompt_grounding, {})
#     if "phrases" in grounding_data:
#         phrases = grounding_data["phrases"]
#     else:
#         phrases = grounding_data.get("labels", [])

#     # tokens_positive_eval: 각 구절에 대해 캡션 내 문자 오프셋을 [[start, start+len(phrase)]] 형태로 저장
#     tokens_positive_eval = []
#     for phrase in phrases:
#         start = caption_text.find(phrase)
#         if start != -1:
#             tokens_positive_eval.append([[start, start + len(phrase)]])
    
#     # 타겟 클래스와 연관된 구절 필터링 (Yes/No Query)
#     filtered_indices = []
#     for i, phrase in enumerate(phrases):
#         query = f"Does the phrase '{phrase}' refer to a {target_class}? Answer yes or no."
#         answer = run_t5_yesno_query(query, t5_tokenizer, t5_model, device)
#         if "yes" in answer:
#             filtered_indices.append(i)
#     if not filtered_indices:
#         return None, None, None, None, None

#     bboxes = grounding_data.get("bboxes", [])
#     filtered_bboxes = [bboxes[i] for i in filtered_indices if i < len(bboxes)]
#     filtered_phrases = [phrases[i] for i in filtered_indices]
    
#     # filtered_positive_tokens: 각 필터링된 구절에 대해 캡션에서 안전하게 추출 (ensure_offset_pair 사용)
#     filtered_positive_tokens = []
#     for phrase in filtered_phrases:
#         start = caption_text.find(phrase)
#         if start == -1:
#             filtered_positive_tokens.append([])
#         else:
#             filtered_positive_tokens.append([start, start + len(phrase)])

#     # coco 이미지 항목 생성 (공식 형식에 맞춤)
#     base_name = os.path.basename(image_path)
#     name_no_ext = os.path.splitext(base_name)[0]
#     try:
#         original_img_id = int(name_no_ext)
#     except:
#         original_img_id = global_image_id

#     coco_image = {
#         "file_name": base_name,
#         "height": image.height,
#         "width": image.width,
#         "id": global_image_id,
#         "original_img_id": original_img_id,
#         "caption": caption_text,
#         "dataset_name": dataset_name,
#         # tokens_negative: 예시로 캡션 전체 범위를 사용
#         "tokens_negative": [[0, len(caption_text)]],
#         "sentence_id": 0,
#         "tokens_positive_eval": tokens_positive_eval
#     }

#     annotations = []
#     for i, bbox in enumerate(filtered_bboxes):
#         ann = {
#             "image_id": global_image_id,
#             "id": i + 1, 
#             "bbox": convert_bbox(bbox),
#             "area": round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 3),
#             "iscrowd": 0,
#             "tokens_positive": [filtered_positive_tokens[i]] if i < len(filtered_positive_tokens) else [],
#             "category_id": category_id
#         }
#         annotations.append(ann)

#     return coco_image, annotations, image, bboxes, filtered_bboxes

# # ----------------------------------------
# # 4. GPU 별 이미지 처리 함수
# # ----------------------------------------
# def process_images_on_gpu(gpu_id, image_tuples, annotated_subdir, target_class, dataset_name, category_id, torch_dtype):
#     device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
#     fl_model, fl_processor, t5_tokenizer, t5_model = load_models(device, torch_dtype)
#     os.makedirs(annotated_subdir, exist_ok=True)

#     coco_images = []
#     annotations_all = []
#     annotated_image_paths = []

#     for image_path, global_id in tqdm(image_tuples, desc=f"GPU {gpu_id}", position=gpu_id, leave=True):
#         result = process_image(
#             image_path, global_image_id=global_id,
#             fl_model=fl_model, fl_processor=fl_processor,
#             t5_tokenizer=t5_tokenizer, t5_model=t5_model,
#             device=device, target_class=target_class,
#             dataset_name=dataset_name, category_id=category_id
#         )
#         if result[0] is None:
#             continue
#         coco_img, ann, image_obj, all_bboxes, filtered_bboxes = result

#         # 좌우 결합된 annotated 이미지 생성 (전체 bbox: 파란색, 필터링된 bbox: 빨간색)
#         annotated_all = image_obj.copy()
#         draw_bboxes(annotated_all, all_bboxes, color="blue", width=2)
#         annotated_filtered = image_obj.copy()
#         draw_bboxes(annotated_filtered, filtered_bboxes, color="red", width=3)
#         combined_annotated = combine_images_horizontally(annotated_all, annotated_filtered)

#         base_name = os.path.basename(image_path)
#         save_path = os.path.join(annotated_subdir, base_name)
#         combined_annotated.save(save_path)
#         annotated_image_paths.append(save_path)

#         coco_images.append(coco_img)
#         annotations_all.extend(ann)

#     return coco_images, annotations_all, annotated_image_paths

# # ----------------------------------------
# # 5. Main 함수 (최종 JSON 생성 및 원본 데이터셋 복사)
# # ----------------------------------------
# def main(args):
#     image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
#     image_files = []
#     for ext in image_extensions:
#         image_files.extend(glob(os.path.join(args.input_dir, ext)))
#     image_files = sorted(image_files)
#     if not image_files:
#         print("입력 이미지가 없습니다.")
#         return

#     if args.sample_ratio < 1.0:
#         sample_count = max(1, int(len(image_files) * args.sample_ratio))
#         image_files = random.sample(image_files, sample_count)

#     image_tuples = [(img_path, idx + 1) for idx, img_path in enumerate(image_files)]
#     num_gpus = args.num_gpus
#     groups = [[] for _ in range(num_gpus)]
#     for i, tup in enumerate(image_tuples):
#         groups[i % num_gpus].append(tup)

#     annotated_subdirs = [os.path.join(args.annotated_dir, f"gpu_{i}") for i in range(num_gpus)]
#     for subdir in annotated_subdirs:
#         os.makedirs(subdir, exist_ok=True)

#     results = []
#     with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
#         futures = []
#         for gpu_id in range(num_gpus):
#             futures.append(executor.submit(
#                 process_images_on_gpu,
#                 gpu_id,
#                 groups[gpu_id],
#                 annotated_subdirs[gpu_id],
#                 args.target_class,
#                 args.dataset_name,
#                 args.category_id,
#                 torch.float32
#             ))
#         for future in concurrent.futures.as_completed(futures):
#             results.append(future.result())

#     all_coco_images = []
#     all_annotations = []
#     all_annotated_paths = []
#     for coco_imgs, annos, ann_paths in results:
#         all_coco_images.extend(coco_imgs)
#         all_annotations.extend(annos)
#         all_annotated_paths.extend(ann_paths)

#     # 전체 어노테이션에 대해 고유한 annotation id 재할당
#     global_ann_id = 0
#     for ann in all_annotations:
#         ann["id"] = global_ann_id
#         global_ann_id += 1

#     categories = [{"supercategory": "object", "id": args.category_id, "name": "object"}]
#     coco_annotation = {
#         "images": all_coco_images,
#         "annotations": all_annotations,
#         "categories": categories
#     }
#     with open(args.output_json, "w", encoding="utf-8") as f:
#         json.dump(coco_annotation, f, ensure_ascii=False, indent=2)

#     # 추가: 원본 데이터셋 디렉토리에서 사용된 이미지들을 복사하여 저장 (JSON에 기록된 이미지와 동일한 이름)
#     copy_dir = os.path.join(args.annotated_dir, "dataset_copy")
#     os.makedirs(copy_dir, exist_ok=True)
#     for img in all_coco_images:
#         src_path = os.path.join(args.input_dir, img["file_name"])
#         dst_path = os.path.join(copy_dir, img["file_name"])
#         if os.path.exists(src_path):
#             shutil.copy(src_path, dst_path)
#     print(f"원본 데이터셋 이미지가 '{copy_dir}'에 복사되었습니다.")

# if __name__ == "__main__":
#     import shutil  # 파일 복사 위해 추가
#     parser = argparse.ArgumentParser(
#         description="여러 이미지에 대해 phrase grounding 및 COCO annotation 생성 (병렬 GPU 처리 지원) 및 원본 데이터셋 복사"
#     )
#     parser.add_argument("--input_dir", type=str, required=True, help="입력 이미지 디렉토리 경로")
#     parser.add_argument("--output_json", type=str, required=True, help="출력 COCO annotation JSON 파일 경로")
#     parser.add_argument("--annotated_dir", type=str, required=True, help="각 GPU별 annotated 이미지 저장할 디렉토리")
#     parser.add_argument("--sample_ratio", type=float, default=1.0, help="샘플링 비율 (예: 0.1 = 10%)")
#     parser.add_argument("--num_gpus", type=int, required=True, help="사용할 GPU 수")
#     parser.add_argument("--target_class", type=str, default="Tank", help="타겟 클래스 (기본: tank)")
#     parser.add_argument("--dataset_name", type=str, default="flickr", help="데이터셋 이름 (예: flickr)")
#     parser.add_argument("--category_id", type=int, default=1, help="카테고리 ID (공식 형식에 맞게)")
#     args = parser.parse_args()

#     main(args)

import os
import json
import torch
import re
import numpy as np
from PIL import Image, ImageDraw
from glob import glob
from tqdm import tqdm
import random
import argparse
import concurrent.futures
import shutil  # 이미지 복사를 위해 추가

from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    T5Tokenizer, 
    T5ForConditionalGeneration
)

# ----------------------------------------
# 1. 모델 및 토크나이저 로드 함수
# ----------------------------------------
def load_models(device, torch_dtype):
    fl_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    fl_processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    t5_model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-large", 
        torch_dtype=torch_dtype
    ).to(device)
    t5_model.eval()
    return fl_model, fl_processor, t5_tokenizer, t5_model

def run_t5_yesno_query(query, t5_tokenizer, t5_model, device):
    raw_inputs = t5_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in raw_inputs.items()}
    outputs = t5_model.generate(**inputs, max_new_tokens=16)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    return answer

# ----------------------------------------
# 2. 문자 오프셋 및 bbox 관련 함수
# ----------------------------------------
def get_word_offsets(text):
    offsets = []
    for match in re.finditer(r'\S+', text):
        offsets.append([match.start(), match.end()])
    return offsets

def ensure_offset_pair(t):
    """
    주어진 t가 단일 int이면 [t, t+1]로, 
    길이가 1인 리스트/튜플이면 [t[0], t[0]+1]로, 
    길이가 2인 경우 그대로 반환합니다.
    그 외에는 None을 반환합니다.
    """
    if isinstance(t, int):
        return [t, t+1]
    elif isinstance(t, (list, tuple)):
        if len(t) == 1:
            return [t[0], t[0] + 1]
        elif len(t) == 2:
            return t
    return None

def safe_parse_tokens(caption, tokens_positive):
    """
    주어진 caption과 tokens_positive의 각 오프셋 정보를
    ensure_offset_pair를 통해 [start, end] 형태로 보정하고,
    해당 부분 문자열을 추출하여 리스트로 반환합니다.
    인덱스 범위를 초과하면 해당 항목은 무시합니다.
    """
    parsed_tokens = []
    for t in tokens_positive:
        pair = ensure_offset_pair(t)
        if pair is None:
            continue
        start, end = pair
        if start < 0 or end > len(caption):
            continue
        parsed_tokens.append(caption[start:end])
    return parsed_tokens

def convert_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)]

def combine_images_horizontally(img1, img2):
    # 두 이미지의 높이를 동일하게 맞춘 후 좌우 결합
    h = max(img1.height, img2.height)
    w = img1.width + img2.width
    combined = Image.new("RGB", (w, h))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    return combined

def draw_bboxes(image, bboxes, color="red", width=3):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline=color, width=width)
    return image

# ----------------------------------------
# 3. 단일 이미지 처리 함수
# ----------------------------------------
def process_image(image_path, global_image_id, fl_model, fl_processor, t5_tokenizer, t5_model, device, 
                  target_class="tank", dataset_name="flickr", category_id=1):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return None, None, None, None, None

    # 캡션 생성 (프롬프트: "<MORE_DETAILED_CAPTION>")
    prompt_caption = "<MORE_DETAILED_CAPTION>"
    inputs_caption = fl_processor(text=prompt_caption, images=image, return_tensors="pt").to(device)
    generated_ids_caption = fl_model.generate(
        input_ids=inputs_caption["input_ids"],
        pixel_values=inputs_caption["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text_caption = fl_processor.batch_decode(generated_ids_caption, skip_special_tokens=False)[0]
    caption_dict = fl_processor.post_process_generation(
        generated_text_caption,
        task=prompt_caption,
        image_size=(image.width, image.height)
    )
    caption_text = caption_dict.get(prompt_caption, "")

    # Phrase Grounding (프롬프트: "<CAPTION_TO_PHRASE_GROUNDING>")
    prompt_grounding = "<CAPTION_TO_PHRASE_GROUNDING>"
    grounding_input_text = prompt_grounding + " " + caption_text
    inputs_grounding = fl_processor(text=grounding_input_text, images=image, return_tensors="pt").to(device)
    generated_ids_grounding = fl_model.generate(
        input_ids=inputs_grounding["input_ids"],
        pixel_values=inputs_grounding["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text_grounding = fl_processor.batch_decode(generated_ids_grounding, skip_special_tokens=False)[0]
    grounding_result = fl_processor.post_process_generation(
        generated_text_grounding,
        task=prompt_grounding,
        image_size=(image.width, image.height)
    )

    grounding_data = grounding_result.get(prompt_grounding, {})
    if "phrases" in grounding_data:
        phrases = grounding_data["phrases"]
    else:
        phrases = grounding_data.get("labels", [])

    # tokens_positive_eval: 각 구절에 대해 캡션 내 문자 오프셋을 [[start, start+len(phrase)]] 형태로 저장
    tokens_positive_eval = []
    for phrase in phrases:
        start = caption_text.find(phrase)
        if start != -1:
            tokens_positive_eval.append([[start, start + len(phrase)]])
    
    # 타겟 클래스와 연관된 구절 필터링 (Yes/No Query)
    filtered_indices = []
    for i, phrase in enumerate(phrases):
        query = f"Does the phrase '{phrase}' refer to a {target_class}? Answer yes or no."
        answer = run_t5_yesno_query(query, t5_tokenizer, t5_model, device)
        if "yes" in answer:
            filtered_indices.append(i)
    if not filtered_indices:
        return None, None, None, None, None

    bboxes = grounding_data.get("bboxes", [])
    filtered_bboxes = [bboxes[i] for i in filtered_indices if i < len(bboxes)]
    filtered_phrases = [phrases[i] for i in filtered_indices]
    
    # filtered_positive_tokens: 각 필터링된 구절에 대해 캡션에서 안전하게 추출 (ensure_offset_pair 사용)
    filtered_positive_tokens = []
    for phrase in filtered_phrases:
        start = caption_text.find(phrase)
        if start == -1:
            filtered_positive_tokens.append([])
        else:
            filtered_positive_tokens.append([start, start + len(phrase)])

    # coco 이미지 항목 생성 (공식 형식에 맞춤)
    base_name = os.path.basename(image_path)  # base_name을 정의
    name_no_ext = os.path.splitext(base_name)[0]
    try:
        original_img_id = int(name_no_ext)
    except:
        original_img_id = global_image_id

    coco_image = {
        "file_name": base_name,
        "height": image.height,
        "width": image.width,
        "id": global_image_id,
        "original_img_id": original_img_id,
        "caption": caption_text,
        "dataset_name": dataset_name,
        "tokens_negative": [[0, len(caption_text)]],
        "sentence_id": 0,
        "tokens_positive_eval": tokens_positive_eval
    }

    annotations = []
    for i, bbox in enumerate(filtered_bboxes):
        ann = {
            "image_id": global_image_id,
            "id": i + 1, 
            "bbox": convert_bbox(bbox),
            "area": round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 3),
            "iscrowd": 0,
            "tokens_positive": [filtered_positive_tokens[i]] if i < len(filtered_positive_tokens) else [],
            "category_id": category_id
        }
        annotations.append(ann)

    return coco_image, annotations, image, bboxes, filtered_bboxes

# ----------------------------------------
# 4. GPU 별 이미지 처리 함수
# ----------------------------------------
def process_images_on_gpu(gpu_id, image_tuples, annotated_subdir, target_class, dataset_name, category_id, torch_dtype):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    fl_model, fl_processor, t5_tokenizer, t5_model = load_models(device, torch_dtype)
    os.makedirs(annotated_subdir, exist_ok=True)

    coco_images = []
    annotations_all = []
    annotated_image_paths = []

    for image_path, global_id in tqdm(image_tuples, desc=f"GPU {gpu_id}", position=gpu_id, leave=True):
        result = process_image(
            image_path, global_image_id=global_id,
            fl_model=fl_model, fl_processor=fl_processor,
            t5_tokenizer=t5_tokenizer, t5_model=t5_model,
            device=device, target_class=target_class,
            dataset_name=dataset_name, category_id=category_id
        )
        if result[0] is None:
            continue
        coco_img, ann, image_obj, all_bboxes, filtered_bboxes = result

        # 좌우 결합된 annotated 이미지 생성 (전체 bbox: 파란색, 필터링된 bbox: 빨간색)
        annotated_all = image_obj.copy()
        draw_bboxes(annotated_all, all_bboxes, color="blue", width=2)
        annotated_filtered = image_obj.copy()
        draw_bboxes(annotated_filtered, filtered_bboxes, color="red", width=3)
        combined_annotated = combine_images_horizontally(annotated_all, annotated_filtered)

        base_name = os.path.basename(image_path)
        save_path = os.path.join(annotated_subdir, base_name)
        combined_annotated.save(save_path)
        annotated_image_paths.append(save_path)

        coco_images.append(coco_img)
        annotations_all.extend(ann)

    return coco_images, annotations_all, annotated_image_paths

# ----------------------------------------
# 5. Main 함수 (최종 JSON 생성 및 원본 데이터셋 복사)
# ----------------------------------------
def main(args):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(args.input_dir, ext)))
    image_files = sorted(image_files)
    if not image_files:
        print("입력 이미지가 없습니다.")
        return

    if args.sample_ratio < 1.0:
        sample_count = max(1, int(len(image_files) * args.sample_ratio))
        image_files = random.sample(image_files, sample_count)

    image_tuples = [(img_path, idx + 1) for idx, img_path in enumerate(image_files)]
    num_gpus = args.num_gpus
    groups = [[] for _ in range(num_gpus)]
    for i, tup in enumerate(image_tuples):
        groups[i % num_gpus].append(tup)

    annotated_subdirs = [os.path.join(args.annotated_dir, f"gpu_{i}") for i in range(num_gpus)]
    for subdir in annotated_subdirs:
        os.makedirs(subdir, exist_ok=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id in range(num_gpus):
            futures.append(executor.submit(
                process_images_on_gpu,
                gpu_id,
                groups[gpu_id],
                annotated_subdirs[gpu_id],
                args.target_class,
                args.dataset_name,
                args.category_id,
                torch.float32
            ))
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    all_coco_images = []
    all_annotations = []
    all_annotated_paths = []
    for coco_imgs, annos, ann_paths in results:
        all_coco_images.extend(coco_imgs)
        all_annotations.extend(annos)
        all_annotated_paths.extend(ann_paths)

    # 전체 어노테이션에 대해 고유한 annotation id 재할당
    global_ann_id = 0
    for ann in all_annotations:
        ann["id"] = global_ann_id
        global_ann_id += 1

    categories = [{"supercategory": "object", "id": args.category_id, "name": "object"}]
    coco_annotation = {
        "images": all_coco_images,
        "annotations": all_annotations,
        "categories": categories
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(coco_annotation, f, ensure_ascii=False, indent=2)

    # 추가: 원본 데이터셋 디렉토리에서 사용된 이미지들을 복사하여 저장 (JSON에 기록된 이미지와 동일한 이름)
    copy_dir = os.path.join(args.annotated_dir, "dataset_copy")
    os.makedirs(copy_dir, exist_ok=True)
    for img in all_coco_images:
        src_path = os.path.join(args.input_dir, img["file_name"])
        dst_path = os.path.join(copy_dir, img["file_name"])
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
    print(f"원본 데이터셋 이미지가 '{copy_dir}'에 복사되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="여러 이미지에 대해 phrase grounding 및 COCO annotation 생성 (병렬 GPU 처리 지원) 및 원본 데이터셋 복사"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="입력 이미지 디렉토리 경로")
    parser.add_argument("--output_json", type=str, required=True, help="출력 COCO annotation JSON 파일 경로")
    parser.add_argument("--annotated_dir", type=str, required=True, help="각 GPU별 annotated 이미지 저장할 디렉토리")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="샘플링 비율 (예: 0.1 = 10%)")
    parser.add_argument("--num_gpus", type=int, required=True, help="사용할 GPU 수")
    parser.add_argument("--target_class", type=str, default="Tank", help="타겟 클래스 (기본: tank)")
    parser.add_argument("--dataset_name", type=str, default="flickr", help="데이터셋 이름 (예: flickr)")
    parser.add_argument("--category_id", type=int, default=1, help="카테고리 ID (공식 형식에 맞게)")
    args = parser.parse_args()

    main(args)
