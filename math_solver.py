"""
Phien ban chay truc tiep voi OpenCV (khong can Streamlit).
Chay: python math_solver.py
"""
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from openai import OpenAI
from PIL import Image
import base64
import io

# ==================== CAU HINH ====================
API_KEY = "API_KEY_CUA_BAN"
CAMERA_INDEX = 0
DRAW_COLOR = (255, 107, 107)  # Do hong BGR
LINE_THICKNESS = 10

# Khoi tao OpenAI client
client = OpenAI(api_key=API_KEY)

# Khoi tao camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(3, 1280)
cap.set(4, 720)

# Khoi tao hand detector
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.7,
    minTrackCon=0.5
)


def get_hand_info(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None


def draw_on_canvas(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    # Gio ngon tro = ve
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), DRAW_COLOR, LINE_THICKNESS)

    # Gio ngon cai = xoa canvas
    elif fingers == [1, 0, 0, 0, 0]:
        canvas[:] = 0

    return current_pos, canvas


def send_to_ai(canvas, fingers):
    if fingers == [0, 0, 1, 1, 1]:
        # Convert canvas to base64 image
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Hãy nhìn hình vẽ này và giải bài toán được vẽ. "
                                "Trả lời bằng tiếng Việt, trình bày rõ ràng từng bư��c giải. "
                                "Nếu là phép tính, hãy tính kết quả. "
                                "Nếu là hình học, hãy phân tích và giải."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    return None


# ==================== VONG LAP CHINH ====================
prev_pos = None
canvas = None
ai_response = ""

print("=" * 50)
print("  GIẢI TOÁN BẰNG CỬ CHỈ TAY - AI")
print("=" * 50)
print("Hướng dẫn:")
print("  - Giơ ngón trỏ: Vẽ")
print("  - Giơ ngón cái: Xóa canvas")
print("  - Giơ 3 ngón (giữa+áp út+út): Gửi AI giải")
print("  - Nhấn 'q' để thoát")
print("=" * 50)

while True:
    success, img = cap.read()
    if not success:
        print("Không thể đọc camera!")
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = get_hand_info(img)

    if info:
        fingers, lmList = info
        prev_pos, canvas = draw_on_canvas(info, prev_pos, canvas)

        result = send_to_ai(canvas, fingers)
        if result:
            ai_response = result
            print("\n--- KẾT QUẢ TỪ AI ---")
            print(ai_response)
            print("---" * 10)
    else:
        prev_pos = None

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    if ai_response:
        lines = ai_response.split('\n')[:3]
        y_offset = 50
        for line in lines:
            cv2.putText(image_combined, line[:80], (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 184, 148), 2)
            y_offset += 30

    cv2.putText(image_combined, "Tro: Ve | Cai: Xoa | 3 ngon: Gui AI | q: Thoat",
               (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (223, 230, 233), 1)

    cv2.imshow("Giai Toan Bang Cu Chi Tay - AI", image_combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
