"""
COCO format JSON utilities
"""

import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def merge_json_results(output_dir, num_gpus):
    """여러 GPU의 COCO JSON 결과를 하나로 병합"""
    combined_imgs, combined_anns, combined_cats = [], [], {}
    iid, aid = 1, 1
    for g in range(num_gpus):
        sub  = os.path.join(output_dir, f"gpu_{g}")
        path = os.path.join(sub, "predictions.json")
        if not os.path.exists(path): continue
        data = json.load(open(path, "r", encoding="utf-8"))
        local_map = {}
        for im in data["images"]:
            old = im["id"]
            im["id"] = iid
            local_map[old] = iid
            combined_imgs.append(im)
            iid += 1
        cat_map = {c["id"]:c["name"] for c in data["categories"]}
        for ann in data["annotations"]:
            ann["image_id"] = local_map[ann["image_id"]]
            nm = cat_map[ann["category_id"]]
            if nm not in combined_cats:
                combined_cats[nm] = len(combined_cats)+1
            ann["category_id"] = combined_cats[nm]
            ann["id"] = aid
            combined_anns.append(ann)
            aid += 1
    cats = [{"id":cid,"name":nm} for nm,cid in combined_cats.items()]
    merged = {"images":combined_imgs, "annotations":combined_anns, "categories":cats}
    outp = os.path.join(output_dir, "predictions_merged.json")
    with open(outp,"w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print("Merged COCO JSON →", outp)

def compute_map(gt_json, pred_json):
    coco_gt = COCO(gt_json)
    with open(pred_json, "r", encoding="utf-8") as f:
        pred = json.load(f)
    for ann in pred["annotations"]:
        pass  # Remove score handling entirely
    coco_dt = coco_gt.loadRes(pred["annotations"])
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return coco_eval.stats[0], coco_eval.stats[1] 