import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_map(gt_json_path, pred_json_path):

    # 1) Load GT / Pred
    coco_gt = COCO(gt_json_path)
    gt_images = coco_gt.dataset["images"]
    gt_img_map = {img["file_name"]: img["id"] for img in gt_images}
    gt_img_ids = set(gt_img_map.values())

    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    # pred 구조 정규화
    if isinstance(pred_data, list):
        preds = pred_data
        pred_images = []
    elif "annotations" in pred_data:
        preds = pred_data["annotations"]
        pred_images = pred_data.get("images", [])
    else:
        preds = pred_data
        pred_images = []

    # 2) file_name 기반 image_id 매칭
    pred_img_map = {}
    if pred_images:
        for img in pred_images:
            if "file_name" in img:
                pred_img_map[img["id"]] = img["file_name"]

    corrected_preds = []
    for ann in preds:
        img_id = ann.get("image_id")
        new_img_id = None

        # file_name 기반으로 매칭 시도
        if img_id in pred_img_map:
            fname = pred_img_map[img_id]
            if fname in gt_img_map:
                new_img_id = gt_img_map[fname]
        elif "file_name" in ann:  # annotation에 file_name이 직접 있을 경우
            fname = ann["file_name"]
            if fname in gt_img_map:
                new_img_id = gt_img_map[fname]

        # 직접적인 id 일치 시도
        elif img_id in gt_img_ids:
            new_img_id = img_id

        # 매칭된 경우만 반영
        if new_img_id is not None:
            ann["image_id"] = new_img_id
            corrected_preds.append(ann)
        else:
            pass  # 매칭 실패 시 제외

    if not corrected_preds:
        raise ValueError("No valid annotations remain after matching. Check that file_name exists in pred.json.")

    print(f"[INFO] {len(corrected_preds)} annotations successfully matched to GT image_ids.")

    # 3) Evaluation
    coco_dt = coco_gt.loadRes(corrected_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 4) Per-category precision (Detectron2 style)
    precisions = coco_eval.eval["precision"]
    cat_ids = coco_gt.getCatIds()
    assert len(cat_ids) == precisions.shape[2]

    eval_results = {}

    print("\n[Per-category AP Results]")
    print(f"{'Category':25s} {'AP':>8s} {'AP@0.5':>8s} {'AP@0.75':>8s}")

    for idx, cat_id in enumerate(cat_ids):
        nm = coco_gt.loadCats(cat_id)[0]
        name = nm["name"]

        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap_all = np.mean(precision) if precision.size else float("nan")

        precision_50 = precisions[0, :, idx, 0, -1]
        precision_50 = precision_50[precision_50 > -1]
        ap50 = np.mean(precision_50) if precision_50.size else float("nan")

        precision_75 = precisions[5, :, idx, 0, -1]
        precision_75 = precision_75[precision_75 > -1]
        ap75 = np.mean(precision_75) if precision_75.size else float("nan")

        print(f"{name:25s} {ap_all:8.3f} {ap50:8.3f} {ap75:8.3f}")
        eval_results[name] = {
            "AP": round(float(ap_all), 3),
            "AP@0.5": round(float(ap50), 3),
            "AP@0.75": round(float(ap75), 3),
        }

    print("\n[Summary]")
    print(f"mAP@[0.5:0.95]: {coco_eval.stats[0]:.4f}")
    print(f"mAP@0.5:        {coco_eval.stats[1]:.4f}")
    print(f"mAP@0.75:       {coco_eval.stats[2]:.4f}")

    return eval_results, coco_eval.stats.tolist()


if __name__ == "__main__":
    gt_json_path = "./updated_gt.json"
    pred_json_path = "./result/OVDusingQwen-Qwen2.5-VL-7B-Instruct_dth0.05_rmraw_roi512_test/predictions_merged.json"

    eval_results, stats = compute_map(gt_json_path, pred_json_path)
