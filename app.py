"""
Ứng dụng Giải Toán bằng Cử Chỉ Tay và Nhập Text - Flask version.
Chạy: python app.py
"""
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from openai import OpenAI
from PIL import Image
from flask import Flask, render_template, Response, request, jsonify
import base64
import io
import threading

app = Flask(__name__)

# ==================== BIEN TOAN CUC ====================
camera_lock = threading.Lock()
cap = None
detector = None
canvas = None
prev_pos = None
ai_response = ""

# Cấu hình mặc định
config = {
    "api_key": "YOUR_API_KEY_HERE", 
    "camera_index": 0,
    "draw_color": (107, 107, 255),  # BGR - đỏ hồng
    "line_thickness": 10,
}

# Khởi tạo OpenAI client với API key mặc định
client = OpenAI(api_key=config["api_key"])


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def init_camera():
    global cap, detector
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(config["camera_index"])
    cap.set(3, 1280)
    cap.set(4, 720)
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


def draw_on_canvas(info, prev_pos, canvas_img):
    fingers, lmList = info
    current_pos = None

    # Giơ ngón trỏ = vẽ
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas_img, tuple(current_pos), tuple(prev_pos),
                 config["draw_color"], config["line_thickness"])

    # Giơ ngón cái = xóa canvas
    elif fingers == [1, 0, 0, 0, 0]:
        canvas_img[:] = 0

    return current_pos, canvas_img


# Lưu lịch sử hội thoại cho từng tab
conversation_history = {
    "gesture": [],
    "text": [],
}

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI thông minh hỗ trợ học tập đa lĩnh vực. "
    "Khi nhận được câu hỏi hoặc hình ảnh, hãy tự động nhận diện đây là môn gì "
    "(Toán, Tin học, Vật lý, Hóa học, Sinh học, Tiếng Anh, Văn học, Lịch sử, Địa lý...) "
    "và trả lời đúng theo chuyên ngành đó. "
    "Ví dụ: nếu ảnh là đề thi Tin học thì giải theo kiến thức Tin học (thuật toán, lập trình, cơ sở dữ liệu...), "
    "KHÔNG giải thành bài toán. "
    "Trả lời bằng tiếng Việt, trình bày rõ ràng từng bước. "
    "KHÔNG dùng LaTeX (không dùng \\, $, \\frac, \\times, \\pi). "
    "Viết công thức bằng ký tự thường: dùng x thay \\times, / thay \\frac, π thay \\pi. "
    "Dùng Markdown: **in đậm** cho kết quả, ### cho tiêu đề, - cho danh sách."
)


def send_canvas_to_ai(canvas_img):
    pil_image = Image.fromarray(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history["gesture"])
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Hãy nhìn hình vẽ này và giải bài tập được vẽ."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1500
    )
    answer = response.choices[0].message.content

    # Lưu lịch sử
    conversation_history["gesture"].append({"role": "user", "content": "Giải bài tập từ hình vẽ"})
    conversation_history["gesture"].append({"role": "assistant", "content": answer})
    # Giới hạn lịch sử 20 tin nhắn
    if len(conversation_history["gesture"]) > 20:
        conversation_history["gesture"] = conversation_history["gesture"][-20:]

    return answer


def send_text_to_ai(text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history["text"])
    messages.append({"role": "user", "content": text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1500
    )
    answer = response.choices[0].message.content

    # Lưu lịch sử
    conversation_history["text"].append({"role": "user", "content": text})
    conversation_history["text"].append({"role": "assistant", "content": answer})
    if len(conversation_history["text"]) > 20:
        conversation_history["text"] = conversation_history["text"][-20:]

    return answer


def generate_frames():
    global canvas, prev_pos, ai_response

    while True:
        if cap is None or not cap.isOpened():
            # Tạo frame đen khi chưa có camera
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, "Dang cho camera...", (400, 360),  # OpenCV khong ho tro Unicode
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        with camera_lock:
            success, img = cap.read()

        if not success:
            continue

        img = cv2.flip(img, 1)

        if canvas is None:
            canvas = np.zeros_like(img)

        info = get_hand_info(img)

        if info:
            fingers, lmList = info
            prev_pos, canvas = draw_on_canvas(info, prev_pos, canvas)

            # Giơ 3 ngón = gửi AI
            if fingers == [0, 0, 1, 1, 1]:
                ai_response = send_canvas_to_ai(canvas)
        else:
            prev_pos = None

        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

        _, buffer = cv2.imencode('.jpg', image_combined)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/config', methods=['POST'])
def update_config():
    global client, canvas, prev_pos
    data = request.json

    if 'api_key' in data and data['api_key']:
        config['api_key'] = data['api_key']
        client = OpenAI(api_key=data['api_key'])

    if 'camera_index' in data:
        config['camera_index'] = int(data['camera_index'])
        init_camera()

    if 'draw_color' in data:
        config['draw_color'] = hex_to_bgr(data['draw_color'])

    if 'line_thickness' in data:
        config['line_thickness'] = int(data['line_thickness'])

    return jsonify({"status": "ok"})


@app.route('/api/clear_canvas', methods=['POST'])
def clear_canvas():
    global canvas, ai_response
    if canvas is not None:
        canvas[:] = 0
    ai_response = ""
    return jsonify({"status": "ok"})


@app.route('/api/get_result', methods=['GET'])
def get_result():
    return jsonify({"result": ai_response})


@app.route('/api/send_text', methods=['POST'])
def api_send_text():
    data = request.json
    text = data.get('text', '')
    if not text.strip():
        return jsonify({"result": "Vui lòng nhập bài tập."})
    if not config['api_key']:
        return jsonify({"result": "Vui lòng nhập API Key trước."})

    try:
        result = send_text_to_ai(text)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"result": f"**Lỗi:** {str(e)}"})


@app.route('/api/send_image', methods=['POST'])
def api_send_image():
    if 'image' not in request.files:
        return jsonify({"result": "Vui lòng chọn ảnh."})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"result": "Vui lòng chọn ảnh."})

    image_bytes = file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime = file.mimetype or "image/png"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history["text"])
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Hãy nhìn ảnh này và giải bài tập có trong ảnh. Nhận diện đúng môn học và trả lời theo chuyên ngành."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}}
        ]
    })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1500
        )
        answer = response.choices[0].message.content

        conversation_history["text"].append({"role": "user", "content": "Giải bài tập từ ảnh"})
        conversation_history["text"].append({"role": "assistant", "content": answer})
        if len(conversation_history["text"]) > 20:
            conversation_history["text"] = conversation_history["text"][-20:]

        return jsonify({"result": answer})
    except Exception as e:
        return jsonify({"result": f"**Lỗi:** {str(e)}"})


@app.route('/api/followup', methods=['POST'])
def api_followup():
    data = request.json
    question = data.get('question', '').strip()
    tab = data.get('tab', 'text')

    if not question:
        return jsonify({"result": "Vui lòng nhập câu hỏi."})

    history_key = tab if tab in conversation_history else "text"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history[history_key])
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1500
        )
        answer = response.choices[0].message.content

        conversation_history[history_key].append({"role": "user", "content": question})
        conversation_history[history_key].append({"role": "assistant", "content": answer})
        if len(conversation_history[history_key]) > 20:
            conversation_history[history_key] = conversation_history[history_key][-20:]

        return jsonify({"result": answer})
    except Exception as e:
        return jsonify({"result": f"**Lỗi:** {str(e)}"})


@app.route('/api/clear_history', methods=['POST'])
def api_clear_history():
    data = request.json
    tab = data.get('tab', 'text')
    if tab in conversation_history:
        conversation_history[tab] = []
    return jsonify({"status": "ok"})


# ==================== MAIN ====================
if __name__ == '__main__':
    init_camera()
    print("=" * 50)
    print("  AI HO TRO HOC TAP - CU CHI TAY & NHAP TEXT")
    print("  Mo trinh duyet: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
