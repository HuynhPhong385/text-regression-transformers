import os
import requests
import streamlit as st
import yaml

CONFIG_PATH = os.getenv("API_CONFIG_PATH", "configs/api_config.yaml")


def load_ui_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("ui", {})
    return {}


ui_config = load_ui_config()
API_URL = os.getenv("API_URL", ui_config.get("api_url", "http://localhost:8000"))

st.set_page_config(page_title="Text Regression Demo", page_icon="📊", layout="centered")
st.title(" Text Regression with Transformers")


def fetch_model_info() -> dict:
    try:
        resp = requests.get(f"{API_URL}/model-info", timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def fetch_health() -> dict:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


with st.sidebar:
    st.subheader("Trạng thái hệ thống")
    health = fetch_health()
    if health.get("status") == "ok":
        if health.get("default_model_loaded"):
            st.success(f"API OK — default model: {health.get('default_model')}")
        else:
            st.warning(f"API chạy nhưng default model '{health.get('default_model')}' chưa load được.")
    else:
        st.error(f"Không kết nối được API tại {API_URL}")
    st.caption(f"API_URL = {API_URL}")

# Bước 1: Review input
st.subheader("1. Nhập văn bản")
text_input = st.text_area("Văn bản cần dự đoán", height=150, placeholder="Nhập nội dung review...")

# Bước 2: Model selector
st.subheader("2. Chọn model")
model_info = fetch_model_info()

if "error" in model_info:
    st.error(f"Không lấy được danh sách model: {model_info['error']}")
    selected_model_key = None
else:
    models = model_info.get("models", {})
    default_key = model_info.get("default_model")
    options = list(models.keys())

    def _format_option(key: str) -> str:
        m = models[key]
        status = "" if m["loaded"] else " chưa sẵn sàng"
        return f"{m['display_name']} ({status})"

    selected_model_key = st.selectbox(
        "Model",
        options=options,
        index=options.index(default_key) if default_key in options else 0,
        format_func=_format_option,
    )

    selected_info = models.get(selected_model_key, {})
    if selected_info and not selected_info.get("loaded"):
        st.warning(f"Model này chưa sẵn sàng: {selected_info.get('error')}")

# Bước 3: Predict -> Sentiment Score 
st.subheader("3. Dự đoán")
if st.button("Predict", type="primary"):
    if not text_input.strip():
        st.warning("Vui lòng nhập văn bản.")
    elif not selected_model_key:
        st.warning("Vui lòng chọn model.")
    else:
        with st.spinner("Đang dự đoán..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text_input, "model_name": selected_model_key},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
                st.metric(label="Sentiment Score [0,1]", value=round(result["score"], 4))
                st.caption(f"Model đã dùng: {result['model_name']}")
            except requests.HTTPError as exc:
                st.error(f"Lỗi từ API: {exc.response.text}")
            except Exception as exc:
                st.error(f"Lỗi kết nối API: {exc}")
