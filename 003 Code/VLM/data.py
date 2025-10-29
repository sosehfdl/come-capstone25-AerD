import json

# 입력 및 출력 파일 경로
input_path = "/workspace/dabin/YOLO-World/data/russia/train/annotation/_annotation.coco.json"
output_path = "/workspace/dabin/YOLO-World/data/russia/train/annotation/test_converted.json"

# 세부 클래스 → 대분류 매핑
category_map = {
    "bm-21": "military_truck",
    "bmd-2": "armoredcar",
    "bmp-1": "armoredcar",
    "bmp-2": "armoredcar",
    "btr-70": "armoredcar",
    "btr-80": "armoredcar",
    "mt-lb": "armoredcar",
    "t-64": "tank",
    "t-72": "tank",
    "t-80": "tank"
}

# 대분류에 대한 고정 ID 부여
new_categories = [
    {"id": 0, "name": "tank"},
    {"id": 1, "name": "armoredcar"},
    {"id": 2, "name": "military_truck"}
]
name_to_new_id = {cat["name"]: cat["id"] for cat in new_categories}

# JSON 로드
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# category_id 변환용: 기존 id → name 매핑
old_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

# annotation category_id 변경
new_annotations = []
for ann in data["annotations"]:
    original_name = old_id_to_name[ann["category_id"]]
    new_class = category_map.get(original_name)
    if new_class is not None:
        ann["category_id"] = name_to_new_id[new_class]
        new_annotations.append(ann)
    else:
        print(f"[경고] 알 수 없는 클래스: {original_name} (annotation id {ann['id']})")

# 최종 JSON 구성
converted_data = {
    "images": data["images"],
    "annotations": new_annotations,
    "categories": new_categories
}

# 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, indent=4)

print(f"✅ 변환 완료: {output_path}")
