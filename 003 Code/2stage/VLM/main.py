"""
Main execution file for military object detection and classification
"""

import os
import argparse
import multiprocessing as mp
from pathlib import Path
import torch
import shutil
import copy
from multiprocessing import Value
from tqdm import tqdm

from GroundingVLM.process import process_on_gpu
from GroundingVLM.coco_utils import merge_json_results
import nltk

if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

mp.set_start_method('spawn', force=True)

def main():
    parser = argparse.ArgumentParser()
    # Detection(1차 탐지) 관련 인자 (mmdet만 지원, 앞쪽에 배치)
    parser.add_argument('--det_config', type=str, required=True, help='mmdet config 파일 경로')
    parser.add_argument('--det_ckpt', type=str, required=True, help='mmdet fine-tuned checkpoint 경로')
    parser.add_argument('--det_score_thr', type=float, default=0.05, help='mmdet detection bbox score threshold')
    parser.add_argument('--det_text', type=str, default='military object.', help='mmdet detection text prompt (ex: "military object.")')
    parser.add_argument("--image_dir",     required=True)
    parser.add_argument("--output_dir",    required=True, help="결과를 저장할 베이스 디렉토리")
    parser.add_argument("--margin_ratio",  type=float, default=0.1)
    parser.add_argument("--min_margin",    type=int,   default=10)
    parser.add_argument("--roi_size",      type=int,   default=512)
    parser.add_argument(
        "--region_method",
        choices=["raw", "mean", "blur", "white"],
        default="raw"
    )
    parser.add_argument(
        "--class_desc_path",
        type=str,
        required=True,
        help="클래스별 설명이 담긴 UTF-8 텍스트 파일 경로"
    )
    parser.add_argument(
        "--info",
        type=str,
        default="",
        help="서브폴더명 뒤에 붙일 추가 설명(예: _experimentA)"
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="오버레이(시각화) 이미지를 저장할지 여부 (기본: 저장 안 함)"
    )
    parser.add_argument(
        "--vl_model",
        type=str,
        default="qwen",
        choices=["qwen", "internvl3"],
        help="분류에 사용할 VL 모델(qwen, internvl3)"
    )
    parser.add_argument(
        "--vl_model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="허깅페이스 VL 모델 경로(Qwen/Qwen2.5-VL-7B-Instruct, OpenGVLab/InternVL3-8B-Instruct 등)"
    )
    parser.add_argument('--contrast_up', action='store_true', help='탐지 전 이미지 대비(히스토그램 평준화) 향상 적용 여부')
    args = parser.parse_args()

    # ── output_dir 아래에 파라미터 기반 서브폴더 생성 ────────────────────
    vlm_tag = f"OVDusing{args.vl_model.capitalize()}"
    model_base = os.path.basename(args.vl_model_id).replace("/", "-")
    model_name = f"{vlm_tag}-{model_base}"
    info_suffix = f"_{args.info}" if args.info else ""
    subname = (
        f"{model_name}"
        f"_dth{args.det_score_thr}"
        f"_rm{args.region_method}"
        f"_roi{args.roi_size}"
        f"{info_suffix}"
    )
    full_output = os.path.join(args.output_dir, subname)
    os.makedirs(full_output, exist_ok=True)
    args.output_dir = full_output
    # ────────────────────────────────────────────────────────────────

    # 클래스 설명 로드
    with open(args.class_desc_path, "r", encoding="utf-8") as f:
        class_descriptions = f.read()

    # 이미지 경로 분할
    img_paths = list(map(str, Path(args.image_dir).rglob("*.[jp][pn]g")))
    ngpu      = torch.cuda.device_count() or 1
    parts     = [img_paths[i::ngpu] for i in range(ngpu)]
    for i, part in enumerate(parts):
        print(f"[할당] GPU {i}에 이미지 {len(part)}장 할당")

    # 전체 이미지 개수
    total_img_count = len(img_paths)
    counter = Value('i', 0)
    # 메인 tqdm
    pbar = tqdm(total=total_img_count, desc="[전체 진행률]", position=0, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    # GPU별 프로세스 실행
    procs = []
    gpu_output_dirs = []
    for gid in range(ngpu):
        gpu_output_dir = os.path.join(args.output_dir, f"gpu_{gid}")
        os.makedirs(gpu_output_dir, exist_ok=True)
        args_gpu = copy.deepcopy(args)
        args_gpu.output_dir = gpu_output_dir
        p = mp.Process(
            target=process_on_gpu,
            args=(gid, parts[gid], args_gpu, class_descriptions, counter, total_img_count)
        )
        p.start()
        procs.append(p)
        gpu_output_dirs.append(gpu_output_dir)
    # 진행률 업데이트 루프
    last = 0
    while any(p.is_alive() for p in procs):
        with counter.get_lock():
            cur = counter.value
        if cur != last:
            pbar.n = cur
            pbar.refresh()
            last = cur
        import time; time.sleep(0.2)
    # 마지막까지
    with counter.get_lock():
        pbar.n = counter.value
        pbar.refresh()
    pbar.close()
    for p in procs:
        p.join()

    # 결과 머지
    merge_json_results(args.output_dir, ngpu)

    # --save_vis 옵션이 켜진 경우, 임시 폴더 삭제 전에 오버레이 이미지 복사
    import glob
    overlaid_dir = os.path.join(args.output_dir, "overlaid")
    os.makedirs(overlaid_dir, exist_ok=True)
    for d in gpu_output_dirs:
        for img_path in glob.glob(os.path.join(d, "*_result.jpg")):
            shutil.copy(img_path, overlaid_dir)

    # 임시 GPU별 predictions.json 및 폴더 삭제
    for d in gpu_output_dirs:
        try:
            shutil.rmtree(d)
        except Exception as e:
            print(f"임시폴더 삭제 실패: {d}, {e}")

    # ── 실행 정보 요약 박스 출력 ──────────────────────────────
    print("\n" + "="*60)
    print("   GroundingVLM Military Object Detection & Classification")
    print("="*60)
    print(f"[모델 종류]   : {args.vl_model}")
    print(f"[모델 ID]     : {args.vl_model_id}")
    print(f"[Detection]   : {args.det_config} (ckpt: {os.path.basename(args.det_ckpt)})")
    print(f"[Score Thresh]: {args.det_score_thr}")
    print(f"[Region Proc] : {args.region_method} (ROI: {args.roi_size})")
    print(f"[클래스 설명] : {args.class_desc_path}")
    print(f"[이미지 폴더] : {args.image_dir}")
    print(f"[전체 이미지]: {len(img_paths)}장")
    print(f"[GPU 사용]    : {ngpu}개")
    print(f"[출력 디렉토리]: {args.output_dir}")
    print("="*60 + "\n")
    # ──────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("[작업 완료] 결과가 저장되었습니다.")
    print(f"[최종 COCO JSON]: {os.path.join(args.output_dir, 'predictions.json')}")
    print(f"[오버레이 이미지]: {os.path.join(args.output_dir, 'overlaid')}")
    print("="*60 + "\n")

if __name__=="__main__":
    main() 