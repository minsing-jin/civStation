import glob
import json
import os
import shutil  # tempfile 청소용
import tempfile

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

ocr = PaddleOCR(
    lang="korean",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def debug_and_get_title(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # 상단 35% crop
    crop = img.crop((0, 0, w, int(h * 0.35)))
    # 3배 확대 (작은 폰트 인식률 향상)
    crop = crop.resize((crop.size[0] * 3, crop.size[1] * 3), Image.BICUBIC)

    tmpdir = tempfile.mkdtemp()

    try:
        # numpy array 입력 시 파일명 매칭 안되므로 주의
        outputs = ocr.predict(np.array(crop), text_rec_score_thresh=0.0)

        titles = []
        for res in outputs:
            # 디버그용 출력 (필요 없으면 주석 처리)
            # res.print()

            # JSON 저장
            res.save_to_json(tmpdir)

            # 가장 최근 생성된 json 파일 읽기
            js_paths = sorted(glob.glob(os.path.join(tmpdir, "*.json")))
            if not js_paths:
                continue

            js_path = js_paths[-1]
            with open(js_path, encoding="utf-8") as f:
                data = json.load(f)

            # [수정 포인트] 'res' 키가 있을 수도, 없을 수도 있음 -> 안전하게 처리
            payload = data.get("res", data)

            rec_texts = payload.get("rec_texts", [])
            rec_scores = payload.get("rec_scores", [])

            # 텍스트와 점수 결합
            for t, s in zip(rec_texts, rec_scores, strict=False):
                if t and float(s) >= 0.6:  # 신뢰도 0.6 이상
                    titles.append(t)

        # '해양', '산업' 처럼 떨어져 있으면 공백으로 합침
        return " ".join(titles).strip()

    finally:
        # 임시 디렉토리 삭제
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    # 실제 파일 경로로 수정해서 테스트
    print(debug_and_get_title("screenshot.png"))
