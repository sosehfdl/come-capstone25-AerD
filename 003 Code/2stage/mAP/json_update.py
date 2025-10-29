import json
import os

def align_gt_to_pred(pred_json_path, gt_json_path, output_json_path):
    """
    예측 JSON(pred)의 카테고리 순서를 기준으로
    GT JSON(gt)의 categories 및 annotations.category_id를 일치시킴.
    """

    # ----------------------------
    # 1. 파일 로드
    # ----------------------------
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    pred_cats = pred_data.get("categories", [])
    gt_cats = gt_data.get("categories", [])
    annotations = gt_data.get("annotations", [])

    print(f"[INFO] Loaded GT categories: {len(gt_cats)}")
    print(f"[INFO] Loaded Pred categories: {len(pred_cats)}")

    # ----------------------------
    # 2. 매핑 준비
    # ----------------------------
    # GT: name -> old_id
    gt_name_to_oldid = {c["name"]: c["id"] for c in gt_cats}

    # Pred: name -> new_id
    pred_name_to_newid = {c["name"]: c["id"] for c in pred_cats}

    # ----------------------------
    # 3. 이름 기준으로 GT의 category_id 재할당 매핑 구성
    # ----------------------------
    old_to_new = {}
    for name, old_id in gt_name_to_oldid.items():
        if name in pred_name_to_newid:
            new_id = pred_name_to_newid[name]
            old_to_new[old_id] = new_id
        else:
            print(f"[WARN] '{name}' not found in prediction categories. Removing this category.")
            old_to_new[old_id] = -1  # 제거 표시

    # ----------------------------
    # 4. GT의 categories를 pred 순서로 정렬
    # ----------------------------
    gt_data["categories"] = [
        {"id": pred_name_to_newid[name], "name": name}
        for name in pred_name_to_newid.keys()
        if name in gt_name_to_oldid
    ]

    # ----------------------------
    # 5. annotations.category_id 갱신
    # ----------------------------
    updated_annotations = []
    for ann in annotations:
        old_cid = ann["category_id"]
        if old_cid in old_to_new:
            new_cid = old_to_new[old_cid]
            if new_cid != -1:
                ann["category_id"] = new_cid
                updated_annotations.append(ann)
            else:
                print(f"[WARN] Removing annotation with old_cid={old_cid} (not in pred).")
        else:
            print(f"[WARN] Unknown category id {old_cid} in annotation. Skipped.")

    gt_data["annotations"] = updated_annotations

    # ----------------------------
    # 6. 결과 저장
    # ----------------------------
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=4)

    print(f"[SUCCESS] Updated GT JSON saved to: {output_json_path}")
    print(f"[INFO] Categories aligned with prediction JSON ({len(gt_data['categories'])} total)")
    print(f"[INFO] Valid annotations: {len(gt_data['annotations'])}")


# ----------------------------
# 사용 예시
# ----------------------------
if __name__ == "__main__":
    pred_json_path = "/mnt/d/py/AIM/Projects/Capstone2025/code/2stage/GroundingVLM/result/OVDusingQwen-Qwen2.5-VL-7B-Instruct_dth0.05_rmraw_roi512_test/predictions_merged.json"
    gt_json_path = "/mnt/d/py/AIM/Projects/Drone_detection/OVD/dataset_datamaker/test_dataset/250526/test.json"
    output_json_path = "./updated_gt.json"

    align_gt_to_pred(pred_json_path, gt_json_path, output_json_path)
