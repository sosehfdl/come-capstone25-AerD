#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO JSON 기반 GT vs Pred 전처리·평가·오버레이 스크립트

1) GT JSON annotation id 재할당
2) 예측 JSON의 category id 및 image/annotation id 통일
3) 예측 JSON의 image/annotation id 갱신
4) mAP, AP@0.5, Per-class Precision 계산 및 출력
5) GT vs Pred bbox 오버레이 이미지 생성
"""
import os
import json
import argparse
import tempfile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def update_gt_annotations(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for new_id, ann in enumerate(data.get("annotations", []), start=1):
        ann["id"] = new_id
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
    json.dump(data, tmp, ensure_ascii=False, indent=4)
    tmp.close()
    return tmp.name


def unify_categories_by_name(gt_json_path, pred_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred = json.load(f)
    gt_map = {c["name"]: c["id"] for c in gt.get("categories", [])}
    old2new = {}
    for cat in pred.get("categories", []):
        new_id = gt_map.get(cat["name"], -1)   # GT의 id를 기준으로 매핑
        old2new[cat["id"]] = new_id
        cat["id"] = new_id
    for ann in pred.get("annotations", []):
        ann["category_id"] = old2new.get(ann["category_id"], -1)

    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
    json.dump(pred, tmp, ensure_ascii=False, indent=4)
    tmp.close()
    return tmp.name


def update_predicted_ids(gt_json_path, pred_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred = json.load(f)
    gt_map = {img["file_name"]: img["id"] for img in gt.get("images", [])}
    new_images, img_id_map = [], {}
    for img in pred.get("images", []):
        fn = img.get("file_name")
        if fn in gt_map:
            img_id_map[img["id"]] = gt_map[fn]
            img["id"] = gt_map[fn]
            new_images.append(img)
    pred["images"] = new_images

    new_anns = []
    for new_ann_id, ann in enumerate(pred.get("annotations", []), start=1):
        old_img = ann["image_id"]
        if old_img in img_id_map:
            ann["image_id"] = img_id_map[old_img]
            ann["id"] = new_ann_id
            new_anns.append(ann)
    pred["annotations"] = new_anns

    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
    json.dump(pred, tmp, ensure_ascii=False, indent=4)
    tmp.close()
    return tmp.name


def compute_map(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred = json.load(f)
    coco_dt = coco_gt.loadRes(pred.get("annotations", []))
    evaler = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaler.evaluate()
    evaler.accumulate()
    evaler.summarize()
    return evaler


def compute_per_class_precision(gt_json_path, pred_json_path, iou_thr=0.5):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred = json.load(f)

    def iou(a, b):
        xa, ya, wa, ha = a
        xb, yb, wb, hb = b
        xi, yi = max(xa, xb), max(ya, yb)
        xa2, ya2 = xa + wa, ya + ha
        xb2, yb2 = xb + wb, yb + hb
        inter = max(0, min(xa2, xb2) - xi) * max(0, min(ya2, yb2) - yi)
        area = wa * ha + wb * hb - inter
        return inter / area if area > 0 else 0

    gt_by, pred_by = {}, {}
    for a in gt["annotations"]:
        gt_by.setdefault(a["image_id"], []).append(a)
    for a in pred["annotations"]:
        pred_by.setdefault(a["image_id"], []).append(a)

    stats = {c["id"]: {"TP": 0, "FP": 0} for c in gt["categories"]}
    for img_id, pr_anns in pred_by.items():
        gts = gt_by.get(img_id, [])
        used = [False] * len(gts)
        for p in pr_anns:
            best, bi = 0, -1
            for i, g in enumerate(gts):
                if used[i] or p["category_id"] != g["category_id"]:
                    continue
                j = iou(p["bbox"], g["bbox"])
                if j > best:
                    best, bi = j, i
            if best >= iou_thr and bi >= 0:
                stats[p["category_id"]]["TP"] += 1
                used[bi] = True
            else:
                stats[p["category_id"]]["FP"] += 1

    names = {c["id"]: c["name"] for c in gt["categories"]}
    for cid, v in stats.items():
        tp, fp = v["TP"], v["FP"]
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        print(f"[Precision] {names[cid]}: {prec:.4f} (TP={tp}, FP={fp})")


def overlay_annotations_on_images(gt_json_path, pred_json_path, img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred = json.load(f)
    imgs = {img["id"]: img["file_name"] for img in gt["images"]}
    gt_by, pred_by = {}, {}
    for a in gt["annotations"]:
        gt_by.setdefault(a["image_id"], []).append(a)
    for a in pred["annotations"]:
        pred_by.setdefault(a["image_id"], []).append(a)

    # 폰트 크기 증가
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 32
        )
        large_font = True
    except:
        font = ImageFont.load_default()
        large_font = False

    # GT 클래스 이름 매핑 만들기
    gt_names = {c["id"]: c["name"] for c in gt["categories"]}

    for iid, fn in tqdm(imgs.items(), desc="Overlay"):
        path = os.path.join(img_dir, fn)
        if not os.path.isfile(path):
            continue
        im = Image.open(path).convert("RGB")
        dr = ImageDraw.Draw(im)

        # GT 객체 그리기
        for a in gt_by.get(iid, []):
            x, y, w, h = a["bbox"]
            cat_id = a.get("category_id", -1)
            class_name = gt_names.get(cat_id, "Unknown")

            # 박스 그리기
            dr.rectangle([x, y, x + w, y + h], outline="green", width=3)

            # 클래스 이름 표시
            gt_label = f"GT: {class_name}"
            text_bbox = font.getbbox(gt_label)
            text_width, text_height = (
                text_bbox[2] - text_bbox[0],
                text_bbox[3] - text_bbox[1],
            )
            padding_x = 2
            padding_y = 1 if not large_font else 4

            # 텍스트 배경
            dr.rectangle(
                [
                    x,
                    y - text_height - 2 * padding_y,  # 박스 위에 텍스트 표시
                    x + text_width + 2 * padding_x,
                    y,
                ],
                fill="green",
            )

            # 텍스트 표시
            dr.text(
                (x + padding_x, y - text_height - padding_y),
                gt_label,
                fill="white",
                font=font,
            )

        # 예측 객체 그리기
        for a in pred_by.get(iid, []):
            x, y, w, h = a["bbox"]
            score = a.get("score", 0.0)
            cat_id = a.get("category_id", -1)
            class_name = next(
                (c["name"] for c in pred["categories"] if c["id"] == cat_id), "Unknown"
            )
            label = f"Pred: {class_name} ({score:.2f})"
            text_bbox = font.getbbox(label)
            text_width, text_height = (
                text_bbox[2] - text_bbox[0],
                text_bbox[3] - text_bbox[1],
            )
            padding_x = 2
            padding_y = 1 if not large_font else 4

            # 박스 그리기
            dr.rectangle([x, y, x + w, y + h], outline="red", width=3)

            # 텍스트 배경
            dr.rectangle(
                [
                    x,
                    y + h - text_height - 2 * padding_y,
                    x + text_width + 2 * padding_x,
                    y + h,
                ],
                fill="red",
            )

            # 텍스트 표시
            dr.text(
                (x + padding_x, y + h - text_height - padding_y),
                label,
                fill="white",
                font=font,
            )

        im.save(os.path.join(output_dir, fn))


def draw_PR_graph(
    evaler,
    save_path_prefix="./pr_curve",
):
    recalls = evaler.params.recThrs
    iou_thr_list = [0.5, 0.75, 0.95]
    color_list = ["tab:blue", "tab:orange", "tab:green"]
    # 1. 각 IoU별 개별 PR 곡선 저장
    for i, iou_thr in enumerate(iou_thr_list):
        iou_thr_idx = np.where(np.isclose(evaler.params.iouThrs, iou_thr))[0][0]
        precision = evaler.eval["precision"][iou_thr_idx, :, 0, 0, 2]
        plt.figure(figsize=(7, 5))
        plt.plot(recalls, precision, label=f"AP@{iou_thr:.2f}", color=color_list[i], linewidth=2)
        plt.xlabel("Recall", fontsize=14)
        plt.ylabel("Precision", fontsize=14)
        plt.title(f"Precision-Recall Curve (IoU={iou_thr:.2f})", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_{str(iou_thr).replace('.', '')}.png")
        plt.close()
    # 2. 한 그래프에 모두 그려서 저장
    plt.figure(figsize=(7, 5))
    for i, iou_thr in enumerate(iou_thr_list):
        iou_thr_idx = np.where(np.isclose(evaler.params.iouThrs, iou_thr))[0][0]
        precision = evaler.eval["precision"][iou_thr_idx, :, 0, 0, 2]
        plt.plot(recalls, precision, label=f"AP@{iou_thr:.2f}", color=color_list[i], linewidth=2)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Precision-Recall Curve (All IoU)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_all.png")
    plt.close()
    print(f"[✔] PR 곡선 저장 완료: {save_path_prefix}_05.png, {save_path_prefix}_075.png, {save_path_prefix}_095.png, {save_path_prefix}_all.png")


def find_failed_images(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred = json.load(f)
    coco_dt = coco_gt.loadRes(pred["annotations"])

    evaler = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaler.evaluate()

    # 1. 이미지별로 GT 개수와 매칭된 detection 수 확인
    no_detections = set()
    partial_miss = set()

    for eval_img in evaler.evalImgs:
        if eval_img is None:
            continue
        img_id = eval_img["image_id"]
        gt_ids = eval_img["gtIds"]
        dt_matches = eval_img["dtMatches"][0]  # IoU threshold @ 0.5 by default
        matched_gts = set(dt_matches[dt_matches > 0])

        if len(dt_matches) == 0:
            no_detections.add(img_id)
        elif len(matched_gts) < len(gt_ids):
            partial_miss.add(img_id)

    print("\n[이미지 전체 탐지 실패]")
    for i in sorted(no_detections):
        print(coco_gt.loadImgs([i])[0]["file_name"])

    print("\n[일부 객체 탐지 실패한 이미지]")
    for i in sorted(partial_miss - no_detections):
        print(coco_gt.loadImgs([i])[0]["file_name"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="GT JSON 경로")
    p.add_argument("--pred", required=True, help="예측 JSON 경로")
    p.add_argument("--imgdir", required=True, help="원본 이미지 디렉토리")
    p.add_argument("--out", default="overlaid", help="오버레이 이미지 저장 디렉토리")
    args = p.parse_args()

    # 1) GT annotation id 재할당
    gt1 = update_gt_annotations(args.gt)
    # 2) pred 카테고리 id 통일
    pred1 = unify_categories_by_name(gt1, args.pred)
    # 3) pred image/ann id 갱신
    pred2 = update_predicted_ids(gt1, pred1)

    # 4) 평가 지표 출력
    print("\n=== COCOeval 결과 ===")
    ev = compute_map(gt1, pred2)
    print(f"Overall mAP (0.50:0.95) = {ev.stats[0]:.4f}")
    print(f"Overall AP@0.5       = {ev.stats[1]:.4f}\n")
    print("=== Per-class Precision (IoU>=0.5) ===")
    compute_per_class_precision(gt1, pred2)

    draw_PR_graph(ev)

    # 5) 오버레이 이미지 생성
    overlay_annotations_on_images(gt1, pred2, args.imgdir, args.out)

    find_failed_images(gt1, pred2)

    # 임시파일 삭제
    for f in (gt1, pred1, pred2):
        try:
            os.remove(f)
        except:
            pass


if __name__ == "__main__":
    main()
