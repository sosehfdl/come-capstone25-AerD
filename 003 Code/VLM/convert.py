# ── 1. MERGE → ZERO-BASED FIX SCRIPT ─────────────────────────────────────────────
import json
import os

# 원본 merged COCO JSON (1-based category_id)
input_merged = "/workspace/dabin/YOLO-World/data/russia/qwen/hard/detection_results_coco_merged.json"
# 여기에 zero-based 로 바꾼 JSON 저장
output_fixed = "/workspace/dabin/YOLO-World/data/russia/qwen/hard/detection_results_coco_zero_based.json"

with open(input_merged, "r", encoding="utf-8") as f:
    data = json.load(f)

# categories: id → id-1
for cat in data["categories"]:
    old = cat["id"]
    cat["id"] = old - 1

# annotations: category_id → category_id-1
for ann in data["annotations"]:
    ann["category_id"] = ann["category_id"] - 1

# 저장
os.makedirs(os.path.dirname(output_fixed), exist_ok=True)
with open(output_fixed, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"[INFO] zero-based COCO JSON saved → {output_fixed}")

import json

# 파일 경로
gt_path = "/workspace/dabin/YOLO-World/data/russia/train/annotation/test_converted.json"
target_path = "/workspace/dabin/YOLO-World/data/russia/qwen/hard/detection_results_coco_zero_based.json"
output_path = "/workspace/dabin/YOLO-World/data/russia/qwen/hard/detection_results_coco_fixed.json"

# target_path = "/workspace/dabin/YOLO-World/data/russia/florence2_prompt2/annotation/military_phrase_grounding.json"
# output_path= "/workspace/dabin/YOLO-World/data/russia/florence2_prompt2/annotation/detection_results_coco_fixed.json"

# GT 파일 로드
with open(gt_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

filename_to_gtid = {img["file_name"]: img["id"] for img in gt_data["images"]}

# 대상 파일 로드
with open(target_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 이미지 ID를 GT 기반으로 매핑
oldid_to_newid = {}
matched = 0

for img in data["images"]:
    fname = img["file_name"]
    old_id = img["id"]
    if fname in filename_to_gtid:
        new_id = filename_to_gtid[fname]
        img["id"] = new_id
        oldid_to_newid[old_id] = new_id
        matched += 1
    else:
        print(f"[WARNING] GT에 없는 파일명: {fname}")

# 어노테이션의 image_id도 업데이트
updated = 0
for ann in data["annotations"]:
    old_img_id = ann["image_id"]
    if old_img_id in oldid_to_newid:
        ann["image_id"] = oldid_to_newid[old_img_id]
        updated += 1
    else:
        print(f"[WARNING] annotations에서 매칭 실패: image_id {old_img_id}")

# 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"[INFO] 이미지 ID 매핑 완료: {matched}개")
print(f"[INFO] 어노테이션 ID 업데이트 완료: {updated}개")
print(f"[INFO] 저장 완료 → {output_path}")


import json

# 파일 경로
input_path = "/workspace/dabin/YOLO-World/data/russia/qwen/hard/detection_results_coco_fixed.json"
output_path = "/workspace/dabin/YOLO-World/data/russia/qwen/hard/prediction.json"

# input_path = "/workspace/dabin/YOLO-World/data/russia/florence2_prompt2/annotation/detection_results_coco_fixed.json"
# output_path = "/workspace/dabin/YOLO-World/data/russia/florence2_prompt2/annotation/prediction.json"

# 파일 로드
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Prediction format 변환 (with ID starting from 0)
predictions = []
for idx, ann in enumerate(data["annotations"]):
    image_id = ann["image_id"]
    category_id =ann["category_id"]
    x, y, w, h = ann["bbox"]
    score = ann.get("score", 1.0)

    predictions.append({
        "id": idx,  # ✅ ID 추가, 0부터 시작
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "score": score
    })

# 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=4)

print(f"[✅] ID 포함 COCO Prediction JSON 저장 완료: {output_path}")

