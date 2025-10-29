# import json
# import argparse

# def update_category_ids(data):
#     """
#     data 내에 'annotations' 혹은 'images' 키가 존재한다면,
#     각 객체의 category_id 값을 변경합니다.
#     기존 category_id가 8이면 새 값은 1 (Tank),
#     그 외에는 새 값은 2 (Armored_car)로 설정합니다.
#     """
#     # annotations 필드가 있을 경우
#     if "annotations" in data:
#         for obj in data["annotations"]:
#             # 기존 category_id 가져오기
#             old_cat_id = obj.get("category_id")
#             if old_cat_id is not None:
#                 if old_cat_id == 8:
#                     obj["category_id"] = 1
#                 else:
#                     obj["category_id"] = 2

#     # images 필드 내에 category_id가 있는 경우도 처리 (만약 존재한다면)
#     if "images" in data:
#         for img in data["images"]:
#             old_cat_id = img.get("category_id")
#             if old_cat_id is not None:
#                 if old_cat_id == 8:
#                     img["category_id"] = 1
#                 else:
#                     img["category_id"] = 2
#     return data

# def main():
#     parser = argparse.ArgumentParser(description="JSON 파일의 category_id 값을 수정합니다.")
#     parser.add_argument("json_path", help="입력 JSON 파일 경로")
#     parser.add_argument("-o", "--output", help="출력 JSON 파일 경로 (지정하지 않으면 입력 파일을 덮어씁니다.)", default=None)
#     args = parser.parse_args()

#     # # JSON 파일 읽기
#     # with open(args.json_path, "r", encoding="utf-8") as f:
#     #     data = json.load(f)
#     # JSON 파일 읽기
#     with open(args.json_path, "r", encoding="utf-8-sig") as f:
#         data = json.load(f)

#     # category_id 업데이트
#     updated_data = update_category_ids(data)

#     # 출력 경로 결정
#     output_path = args.output if args.output else args.json_path

#     # 결과 JSON 저장
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(updated_data, f, ensure_ascii=False, indent=4)


#     print(f"수정된 JSON 파일이 '{output_path}'에 저장되었습니다.")

# if __name__ == "__main__":
#     main()

####################################################################
import json

file_path = "data/tank/tank_json/tank_base_text.json"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("JSON 파일이 정상적으로 로드됨.")
except json.JSONDecodeError as e:
    print(f"JSON 파일 오류: {e}")

####################################################################
import codecs
import shutil

input_file = "data/tank/tank_json/tank_base_text.json"
temp_file = "data/tank/tank_json/tank_base_text.json"

# BOM 제거 후 새로운 파일로 저장
with codecs.open(input_file, "r", "utf-8-sig") as f:
    content = f.read()

with codecs.open(temp_file, "w", "utf-8") as f:
    f.write(content)

# 원본 파일을 백업하고 새 파일을 원래 이름으로 변경
shutil.move(temp_file, input_file)

print("BOM 제거 완료")