"""
GPU-based processing logic for military object detection and classification
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
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
from GroundingVLM.grounding_utils import run_dino, run_sam_mask, preprocess_roi, classify_roi, histogram_equalize, run_mmdet_grounding_dino_inference
import traceback

logging.set_verbosity_error()

def process_on_gpu(gpu_id, image_files, args, class_descriptions, counter, total_img_count):
    """단일 GPU에서 이미지 처리를 수행하는 함수 (진행률은 메인 프로세스에서 통합 관리)"""
    import importlib
    print(f"[DEBUG][GPU{gpu_id}] save_vis: {getattr(args, 'save_vis', None)}")
    print(f"[DEBUG][GPU{gpu_id}] output_dir: {getattr(args, 'output_dir', None)}")
    device    = f"cuda:{gpu_id}"
    dtype     = torch.float16 if torch.cuda.is_available() else torch.float32

    # SAM 모델 로드 (한 번)
    sam_proc  = SamProcessor.from_pretrained("facebook/sam-vit-huge", use_fast=True)
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()

    # VL 모델 로드 (Qwen/InternVL3)
    if args.vl_model == "qwen":
        q_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.vl_model_id, torch_dtype=dtype, device_map={"":device}, trust_remote_code=True
        )
        q_proc  = AutoProcessor.from_pretrained(args.vl_model_id, trust_remote_code=True, use_fast=True)
        q_tokenizer = AutoTokenizer.from_pretrained(args.vl_model_id, trust_remote_code=True)
        vl_model = q_model
        vl_processor = q_proc
        vl_tokenizer = q_tokenizer
    elif args.vl_model == "internvl3":
        import math
        def split_model(model_path):
            device_map = {}
            world_size = torch.cuda.device_count()
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            num_layers = config.llm_config.num_hidden_layers
            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for j in range(num_layer):
                    device_map[f'language_model.model.layers.{layer_cnt}'] = i
                    layer_cnt += 1
            device_map['vision_model'] = 0
            device_map['mlp1'] = 0
            device_map['language_model.model.tok_embeddings'] = 0
            device_map['language_model.model.embed_tokens'] = 0
            device_map['language_model.output'] = 0
            device_map['language_model.model.norm'] = 0
            device_map['language_model.model.rotary_emb'] = 0
            device_map['language_model.lm_head'] = 0
            device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
            return device_map
        device_map = split_model(args.vl_model_id)
        vl_model = AutoModel.from_pretrained(
            args.vl_model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
        vl_tokenizer = AutoTokenizer.from_pretrained(args.vl_model_id, trust_remote_code=True, use_fast=False)
        vl_processor = None  # InternVL3는 별도 processor 없음
    else:
        raise ValueError(f"지원하지 않는 VL 모델: {args.vl_model}")

    font     = ImageFont.load_default()
    results  = []

    # mmdet 기반 detection: 각 이미지별로 bbox 추출
    for idx, path in enumerate(image_files, 1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[GPU{gpu_id}] 이미지 로딩 실패: {path} ({e})")
            continue
        w, h = img.size
        base = Path(path).stem
        # contrast_up 옵션 적용: 탐지 전 이미지 대비 향상
        if getattr(args, 'contrast_up', False):
            img_proc = histogram_equalize(img)
        else:
            img_proc = img
        # 1. DINO bbox 추출 (좌표만 저장)
        det_result = run_mmdet_grounding_dino_inference(
            config_path=args.det_config,
            checkpoint_path=args.det_ckpt,
            image_path=path,
            output_dir=args.output_dir,
            text_prompt=args.det_text,
            score_thr=args.det_score_thr,
            device=device,
            return_results=True
        )
        if isinstance(det_result, list):
            det_result = det_result[0] if det_result else None
        boxes = []
        scores = []
        if det_result and 'predictions' in det_result:
            for pred in det_result['predictions']:
                bboxes = pred.get('bboxes', [])
                scores_list = pred.get('scores', [])
                labels = pred.get('labels', [])
                for bbox, score, label in zip(bboxes, scores_list, labels):
                    if score is not None and score >= args.det_score_thr:
                        x1, y1, x2, y2 = bbox
                        boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                        scores.append(score)
        print(f"[DINO bbox][GPU{gpu_id}] {path} → {boxes}")
        print(f"[DINO score][GPU{gpu_id}] {path} → {scores}")
        if len(boxes) == 0:
            print(f"[경고][GPU{gpu_id}] bbox 없음: {path}")
            continue
        # 1. DINO에서 필터링된 bbox 좌표만 json(임시) 파일에 저장 (클래스명은 비워둠)
        bbox_json_path = os.path.join(args.output_dir, f"{base}_bboxes.json")
        bbox_json_data = []
        for box, score in zip(boxes, scores):
            bbox_json_data.append({
                "bbox": box,
                "category": None,
                "score": score
            })
        with open(bbox_json_path, "w") as f:
            json.dump(bbox_json_data, f, ensure_ascii=False, indent=2)

        # 2. json을 읽어서 region을 crop하고 Qwen 분류기로 클래스명을 얻어 json의 빈 클래스란을 채움
        with open(bbox_json_path, "r") as f:
            bbox_json_data = json.load(f)
        for item in bbox_json_data:
            box = item["bbox"]
            x, y, ww, hh = box
            raw_roi = img.crop((x, y, x+ww, y+hh))
            if args.region_method != "raw":
                with torch.inference_mode():
                    full_mask = run_sam_mask(img, [box], sam_proc, sam_model, device)
                mask_roi = full_mask[y:y+hh, x:x+ww]
                roi      = preprocess_roi(raw_roi, mask_roi, args.region_method)
            else:
                roi = raw_roi
            with torch.inference_mode():
                label = classify_roi(
                    roi, vl_model, vl_processor, args.roi_size, class_descriptions,
                    vl_tokenizer=vl_tokenizer, vl_model_type=args.vl_model
                )
            item["category"] = label
        # Qwen 결과로 채워진 json을 다시 저장
        with open(bbox_json_path, "w") as f:
            json.dump(bbox_json_data, f, ensure_ascii=False, indent=2)

        # 3. bbox+클래스명이 채워진 json을 기반으로 복사본 이미지에 bbox+label을 표시하고 저장
        def visualize_and_save_from_json(img, bbox_json_path, save_path, font):
            with open(bbox_json_path, "r") as f:
                bbox_json_data = json.load(f)
            vis = img.copy()
            draw = ImageDraw.Draw(vis)
            for item in bbox_json_data:
                box = item["bbox"]
                label = item["category"]
                x, y, ww, hh = box
                text = f"{label}"
                bb       = font.getbbox(text)
                tw, th   = bb[2]-bb[0], bb[3]-bb[1]
                draw.rectangle([x, y, x+ww, y+hh], outline="red", width=2)
                draw.rectangle([x, y-th-6, x+tw+4, y], fill="black")
                draw.text((x+2, y-th-4), text, fill="red", font=font)
            vis.save(save_path)

        if len(bbox_json_data) > 0:
            save_path = os.path.join(args.output_dir, f"{base}_result.jpg")
            print(f"[저장 시도][GPU{gpu_id}] {os.path.abspath(save_path)}", flush=True)
            try:
                visualize_and_save_from_json(img, bbox_json_path, save_path, font)
                print(f"[저장 성공][GPU{gpu_id}] {os.path.abspath(save_path)}", flush=True)
            except Exception as e:
                print(f"[저장 실패][GPU{gpu_id}] {os.path.abspath(save_path)}, 에러: {e}", flush=True)
                traceback.print_exc()
        results.append({
            "file_name":   base + Path(path).suffix,
            "width":       w,
            "height":      h,
            "annotations": bbox_json_data
        })
        with counter.get_lock():
            counter.value += 1

    # COCO JSON 쓰기
    imgs, anns, cat2id = [], [], {}
    aid, iid = 1, 1
    for res in results:
        imgs.append({
            "id":       iid,
            "file_name":res["file_name"],
            "width":    res["width"],
            "height":   res["height"]
        })
        for ann in res["annotations"]:
            lbl = ann["category"]
            if lbl not in cat2id:
                cat2id[lbl] = len(cat2id) + 1
            cid = cat2id[lbl]
            anns.append({
                "id":          aid,
                "image_id":    iid,
                "category_id": cid,
                "bbox":        ann["bbox"],
                "area":        ann["bbox"][2] * ann["bbox"][3],
                "iscrowd":     0,
                "score":       ann.get("score", 0)
            })
            aid += 1
        iid += 1

    cats = [{"id": cid, "name": nm} for nm, cid in cat2id.items()]
    out  = {
        "images":      imgs,
        "annotations": anns,
        "categories":  cats
    }
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2) 
