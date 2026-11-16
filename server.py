import io
import json
import base64
import numpy as np
from PIL import Image
import onnxruntime as ort
from http.server import HTTPServer, BaseHTTPRequestHandler

# 1. 載入微調 93.3% 最佳模型與精確字元表
MODEL_PATH = "/Users/linketing/dddd_trainer/projects/tixcraft/models/tixcraft_175_16000.onnx"
CHARSET = [' ', 'k', 'f', 'o', 'n', 'p', 'l', 'm', 'j', 't', 'c', 'u', 'e', 'a', 'g', 'r', 'x', 'z', 'v', 'q', 'h', 'i', 'b', 'd', 'y', 'w', 's']

print("⏳ 正在載入 ONNX OCR 模型...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("✅ 模型載入完成！")

def predict_captcha(img: Image.Image) -> str:
    # 強制縮放為 150x40 灰階
    img = img.convert("L").resize((150, 40))
    # 原生精確標準化: (x / 255.0 - 0.456) / 0.224
    arr = (np.array(img, dtype=np.float32) / 255.0 - 0.456) / 0.224
    tensor = np.expand_dims(arr, axis=(0, 1))

    preds = session.run(["output"], {input_name: tensor})[0]
    raw_idx = preds[0].tolist() if len(preds.shape) == 2 else np.argmax(preds, axis=-1)[0].tolist()

    text, prev = [], -1
    for idx in raw_idx:
        if idx != 0 and idx != prev:
            if idx < len(CHARSET):
                text.append(CHARSET[idx])
        prev = idx
    return "".join(text).lower()

class CaptchaHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            # 支援 JSON 格式 { "image": "data:image/png;base64,..." }
            if post_data.startswith(b'{'):
                body = json.loads(post_data.decode('utf-8'))
                b64_str = body.get('image', body.get('img', ''))
                if ',' in b64_str:
                    b64_str = b64_str.split(',')[1]
                img_bytes = base64.b64decode(b64_str)
            elif post_data.startswith(b'data:image') or b'base64' in post_data[:30]:
                raw_str = post_data.decode('utf-8')
                b64_str = raw_str.split(',')[1] if ',' in raw_str else raw_str
                img_bytes = base64.b64decode(b64_str)
            else:
                img_bytes = post_data

            img = Image.open(io.BytesIO(img_bytes))
            result = predict_captcha(img)

            self.send_response(200)
            self._set_cors_headers()
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            # 拓元助手腳本讀取 data.result
            resp = json.dumps({'code': 200, 'result': result})
            self.wfile.write(resp.encode('utf-8'))
            print(f"🎯 [OCR 辨識成功] 結果: {result}")
        except Exception as e:
            self.send_response(500)
            self._set_cors_headers()
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'code': 500, 'error': str(e), 'result': ''}).encode('utf-8'))
            print(f"❌ [OCR 辨識失敗]: {e}")

    def log_message(self, format, *args):
        pass

if __name__ == '__main__':
    port = 8000
    server = HTTPServer(('127.0.0.1', port), CaptchaHandler)
    print(f"\n🚀 Tixcraft OCR 專用伺服器已就緒！")
    print(f"🌐 監聽位址: http://127.0.0.1:{port}/ocr")
    print(f"🎫 請保持此終端機視窗開啟，瀏覽器即可全自動秒填秒送。\n")
    server.serve_forever()
