"""
Core utility functions for Grounding DINO and Qwen integration
"""

import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    SamProcessor,
    SamModel,
    AutoModelForZeroShotObjectDetection,
    logging,
    AutoModel,
    AutoConfig
)

from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from typing import List, Optional
from mmdet.apis import DetInferencer
import nltk

def letterbox(img: Image.Image, size=(1024,1024)):
    """
    SAM 입력용 1024×1024 레터박스 변환
    """
    tw, th = size
    w, h   = img.size
    scale  = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    pad_x, pad_y = (tw-nw)//2, (th-nh)//2
    resized = img.resize((nw,nh), Image.LANCZOS)
    canvas  = Image.new("RGB", size, (128,128,128))
    canvas.paste(resized, (pad_x,pad_y))
    return canvas, scale, pad_x, pad_y, nw, nh

def run_dino(
    img: Image.Image,
    dth: float,
    tth: float,
    device: str,
    detector_text: list[str],
    proc=None,
    model=None,
):
    """Grounding DINO로 사용자 지정 텍스트(detector_text) 바운딩 박스 검출 (processor/model을 인자로 받아 재사용)"""
    if proc is None:
        proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base", use_fast=True)
    if model is None:
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(device).eval()

    inputs = proc(images=img, text=[detector_text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)

    res = proc.post_process_grounded_object_detection(
        out, inputs.input_ids,
        threshold=dth,
        text_threshold=tth,
        target_sizes=[img.size[::-1]],
    )[0]

    boxes = []
    for b in res["boxes"]:
        x0, y0, x1, y1 = b.tolist()
        boxes.append([int(x0), int(y0), int(x1-x0), int(y1-y0)])
    return boxes

def run_sam_mask(orig: Image.Image, boxes: list, processor: SamProcessor, model: SamModel, device: str):
    """
    원본 이미지에 대해 DINO 박스 리스트로 SAM을 돌린 후 union 마스크 반환
    returns: boolean numpy mask shape (H,W)
    """
    lb, scale, px, py, nw, nh = letterbox(orig)
    Hlb, Wlb = lb.size[1], lb.size[0]
    union = np.zeros((Hlb, Wlb), dtype=bool)

    for (x,y,w_box,h_box) in boxes:
        x1 = float(x*scale + px)
        y1 = float(y*scale + py)
        x2 = float((x+w_box)*scale + px)
        y2 = float((y+h_box)*scale + py)

        inputs = processor(
            lb,
            input_boxes=[[[x1,y1,x2,y2]]],
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        arr = masks[0]  # batch=1
        if torch.is_tensor(arr):
            arr = arr.cpu().numpy()
        collapse_axes = tuple(range(arr.ndim-2))
        bin_mask = (arr > 0.5).any(axis=collapse_axes)
        union |= bin_mask

    crop = union[py:py+nh, px:px+nw]
    crop_u8 = (crop.astype(np.uint8)*255)
    pil_crop = Image.fromarray(crop_u8, mode="L")
    final = np.array(pil_crop.resize(orig.size, resample=Image.NEAREST)) > 0
    return final

def preprocess_roi(raw_roi: Image.Image, mask: np.ndarray, method: str) -> Image.Image:
    """
    raw_roi:   ROI crop (PIL RGB)
    mask:      boolean numpy array [H, W]
    method:    "raw" | "mean" | "blur" | "white"
    """
    m = np.array(mask)
    m = np.squeeze(m)
    if m.ndim > 2:
        m = m[..., 0]
    mask_bool = m.astype(bool)
    H, W = mask_bool.shape

    if method == "raw":
        return raw_roi

    if method == "mean":
        arr = np.array(raw_roi)
        bg = arr[~mask_bool]
        if bg.size == 0:
            color = (128, 128, 128)
        else:
            mc = bg.mean(axis=0).astype(np.uint8)
            color = tuple(int(x) for x in mc)
        background = Image.new("RGB", (W, H), color)

    elif method == "blur":
        background = raw_roi.filter(ImageFilter.GaussianBlur(radius=2))

    elif method == "white":
        background = Image.new("RGB", (W, H), (255, 255, 255))

    else:
        raise ValueError(f"Unknown region_method: {method}")

    mask_img = Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L")
    return Image.composite(raw_roi, background, mask_img)

def letterbox_image(image: Image.Image, target_size=(512,512)):
    """
    Qwen 분류 입력용 레터박스 변환
    """
    tw, th = target_size
    ow, oh = image.size
    scale  = min(tw/ow, th/oh)
    nw, nh = int(ow*scale), int(oh*scale)
    r = image.resize((nw,nh), Image.LANCZOS)
    canvas = Image.new("RGB",(tw,th),(128,128,128))
    px, py = (tw-nw)//2, (th-nh)//2
    canvas.paste(r, (px,py))
    return canvas, scale, px, py

def inference(image: Image.Image, prompt: str, model, processor):
    msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
    chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    vis, vid = process_vision_info(msgs)
    inputs = processor(text=[chat], images=vis, videos=vid,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=512, num_beams=4,
                              do_sample=False, early_stopping=True)
    trimmed = [g[len(i):] for i,g in zip(inputs.input_ids, gen)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

def expand_box(box, img_w, img_h, ratio, min_px):
    x,y,w,h = box
    dx,dy = max(int(w*ratio),min_px), max(int(h*ratio),min_px)
    x0,y0 = max(x-dx,0), max(y-dy,0)
    x1,y1 = min(x+w+dx,img_w), min(y+h+dy,img_h)
    return [x0,y0,x1-x0,y1-y0]

def internvl3_preprocess(roi: Image.Image):
    """
    InternVL3-8B 모델 입력용 공식 전처리 (448x448, max_num=12, 블록 분할)
    """
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    image_size = 448
    max_num = 12
    def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    transform = build_transform(input_size=image_size)
    # 공식 dynamic_preprocess 없이 단일 블록(448x448)만 사용 (ROI는 대부분 1개)
    image = roi.resize((image_size, image_size))
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values

def internvl3_inference(roi: Image.Image, prompt: str, model, tokenizer):
    pixel_values = internvl3_preprocess(roi)
    pixel_values = pixel_values.to(next(model.parameters()).device, dtype=torch.bfloat16)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    question = f"<image>\n{prompt}"
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response

def classify_roi(roi: Image.Image, model, processor, roi_size, class_descriptions, vl_tokenizer=None, vl_model_type="qwen"):
    """
    ROI 이미지를 VL 모델로 분류 (Qwen/InternVL3 모두 지원)
    InternVL3는 roi_size 인자 무시, 448x448 고정
    """
    if vl_model_type == "qwen":
        lb, _, _, _ = letterbox_image(roi, (roi_size, roi_size))
        prompt = (
            "First, determine whether the provided image region shows a general ground-level view or a top-down drone perspective. "
            "Then, analyze the military object based solely on its visual features. "
            "The provided image comes from a detection model and may include both general-view and drone-view angles. "
            "Some objects may be partially damaged. Focus primarily on the most prominent external features described in the provided class list. "
            "From the following class list and descriptions, select the class that best matches the object's distinctive characteristics:\n\n"
            f"{class_descriptions}\n\n"
            "Output only a JSON object in the format {\"label\":\"<class>\"} without any additional text."
        )
        out = inference(lb, prompt, model, processor)
        print("[Qwen inference raw output]", out)
        print("[Qwen inference output type]", type(out))
        # 클래스명만 추출 (빈 줄, 주석, 설명 제외)
        classes = [l.split(":")[0].strip() for l in class_descriptions.strip().split("\n") if l.strip() and not l.startswith("#")]
        try:
            result = json.loads(out)
            print("[Qwen parsed label]", result.get("label", None))
            print("[Qwen class list]", classes)
            print("[Qwen result type]", type(result))
            if result.get("label", None) not in classes:
                print("[Qwen label not in class list, fallback]")
                raise ValueError
        except Exception as e:
            print("[Qwen inference exception, fallback]", e)
            label = classes[0]
            result = {"label": label}
        return result.get("label", None)
    elif vl_model_type == "internvl3":
        prompt = (
            "First, determine whether the provided image region shows a general ground-level view or a top-down drone perspective. "
            "Then, analyze the military object based solely on its visual features. "
            "The provided image comes from a detection model and may include both general-view and drone-view angles. "
            "Some objects may be partially damaged. Focus primarily on the most prominent external features described in the provided class list. "
            "From the following class list and descriptions, select the class that best matches the object's distinctive characteristics:\n\n"
            f"{class_descriptions}\n\n"
            "Output only a JSON object in the format {\"label\":\"<class>\"} without any additional text."
        )
        out = internvl3_inference(roi, prompt, model, vl_tokenizer)
        # 클래스명만 추출 (빈 줄, 주석, 설명 제외)
        classes = [l.split(":")[0].strip() for l in class_descriptions.strip().split("\n") if l.strip() and not l.startswith("#")]
        try:
            result = json.loads(out)
            label = result.get("label", None)
            if label not in classes:
                raise ValueError
        except Exception:
            label = classes[0]
            result = {"label": label}
        return label
    else:
        raise ValueError(f"지원하지 않는 VL 모델 타입: {vl_model_type}")

def histogram_equalize(img: Image.Image) -> Image.Image:
    """
    PIL 이미지를 히스토그램 평준화하여 대비를 높여 반환
    """
    return ImageOps.equalize(img)

def run_mmdet_grounding_dino_inference(
    config_path: str,
    checkpoint_path: str,
    image_path: str,  # 단일 이미지 경로로 변경
    output_dir: str,
    text_prompt: str = "military object.",
    score_thr: float = 0.3,
    # score_thr: bbox score threshold (score 없는 bbox는 region 분류에서 제외)
    device: str = "cuda",
    return_results: bool = False
) -> Optional[List[dict]]:
    """
    MMDetection DetInferencer 기반 Grounding DINO 추론 함수.
    Args:
        config_path: MMDetection config 파일 경로
        checkpoint_path: fine-tuned Grounding DINO .pth 파일 경로
        image_path: 입력 이미지 경로 (단일 파일)
        output_dir: 결과 이미지 저장 디렉토리
        text_prompt: 텍스트 프롬프트(탐지할 클래스)
        score_thr: bbox score threshold
        device: 'cuda' or 'cpu'
        return_results: True면 결과 리스트 반환
    Returns:
        (Optional) DetInferencer의 결과 리스트
    """
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

    os.makedirs(output_dir, exist_ok=True)
    inferencer = DetInferencer(
        model=config_path,
        weights=checkpoint_path,
        device=device if torch.cuda.is_available() else 'cpu'
    )
    result = inferencer(
        image_path,
        texts=text_prompt,
        pred_score_thr=score_thr,
        out_dir=output_dir,
        return_vis=False,
        no_save_pred=True,
        no_save_vis=True
    )
    if return_results:
        return [result]