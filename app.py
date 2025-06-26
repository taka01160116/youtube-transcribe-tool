import streamlit as st
import whisper
import yt_dlp
import subprocess
import os
import uuid
import re
import time
import torch
from glob import glob

st.set_page_config(page_title="YouTube文字起こしツール")

# GPU有無で自動切替
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

@st.cache_resource(show_spinner="Whisperモデルを読み込み中…")
def load_model(model_size):
    model = whisper.load_model(model_size)
    return model.to(DEVICE)

def is_silent_audio(file_path, threshold_db=-40):
    result = subprocess.run(
        ["ffmpeg", "-i", file_path, "-af", "volumedetect", "-f", "null", "-"],
        stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    match = re.search(r"mean_volume: (-?\d+\.?\d*) dB", result.stderr)
    return float(match.group(1)) < threshold_db if match else True

def download_and_convert(url, temp_id):
    m4a_path = f"{temp_id}.m4a"
    wav_path = f"{temp_id}.wav"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': m4a_path,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", m4a_path, "-ar", "16000", "-ac", "1", wav_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"{wav_path} が作成されませんでした。\nffmpeg stderr:\n{result.stderr.decode()}")
    return wav_path, m4a_path

def split_audio_fast(input_file, chunk_length=900):
    output_template = f"{input_file}_part_%03d.wav"
    cmd = [
        "ffmpeg", "-i", input_file,
        "-f", "segment",
        "-segment_time", str(chunk_length),
        "-c", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_template
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return sorted(glob(f"{input_file}_part_*.wav"))

def format_text_japanese(raw_text):
    text = re.sub(r'(?<=[。！？])', '\n', raw_text)
    text = re.sub(r'([^\n]{20,40}?)(が|ので|けど|のに|そして|また|つまり)', r'\1、\2', text)
    return re.sub(r'\n{2,}', '\n', text).strip()

# --- UI ---
st.title("🎙️ YouTube文字起こしツール（完全無料公開版）")
url = st.text_input("🎥 YouTube動画のURLを入力してください：")
model_size = st.selectbox("⚙️ 使用するWhisperモデルを選択：", ["tiny", "base", "medium"], index=1)

st.subheader("📝 整形済み文字起こし")
output_placeholder = st.empty()
copy_btn_placeholder = st.empty()

if st.button("▶️ 文字起こし開始"):
    if not url:
        st.error("まず URL を入力してください")
    else:
        status = st.empty()
        progress_bar = st.progress(0, text="開始準備中…")
        try:
            model = load_model(model_size)
            temp_id = str(uuid.uuid4())

            status.info("🔄 音声ダウンロード中…")
            wav_file, m4a_file = download_and_convert(url, temp_id)

            status.info("🔄 音声分割中…")
            chunks = split_audio_fast(wav_file)
            total_chunks = len(chunks)
            if total_chunks == 0:
                raise RuntimeError("音声の分割に失敗しました。")

            status.success(f"✅ {total_chunks} チャンクに分割されました")

            texts = []
            durations = []

            for i, chunk in enumerate(chunks):
                chunk_size = os.path.getsize(chunk)
                st.write(f"🔍 処理中: {chunk}（{chunk_size} バイト）")

                if chunk_size < 1000:
                    st.warning(f"{chunk} は空のためスキップされました。")
                    continue
                if is_silent_audio(chunk):
                    st.warning(f"{chunk} は無音のためスキップされました。")
                    continue

                try:
                    start = time.time()
                    est = ""
                    if durations:
                        avg = sum(durations) / len(durations)
                        remaining = int(avg * (total_chunks - i))
                        mins, secs = divmod(remaining, 60)
                        est = f"（残り：約 {mins}分 {secs}秒）"
                    progress_bar.progress(min((i + 1) / total_chunks, 1.0), text=f"🧠 {i+1}/{total_chunks} チャンク文字起こし中… {est}")

                    result = model.transcribe(chunk, language="ja", fp16=USE_GPU)["text"]
                    texts.append(result)
                    durations.append(time.time() - start)
                    st.success(f"✅ {chunk} 完了")
                except Exception as e:
                    st.error(f"❌ {chunk} の文字起こしに失敗: {e}")

            progress_bar.progress(1.0, text="🎉 文字起こし完了！")

            full = "\n".join(texts)
            formatted_text = format_text_japanese(full)

            if not formatted_text.strip():
                status.warning("⚠️ 有効な音声が検出されませんでした。")
            else:
                status.success("✅ 全工程が完了しました")
                output_placeholder.text_area("以下が文字起こしの全文です：", formatted_text, height=400)
                copy_btn_placeholder.download_button("📋 全文コピー（テキストファイル）", formatted_text, file_name="transcription.txt")

            # クリーンアップ
            for f in [wav_file, m4a_file] + chunks:
                if os.path.exists(f):
                    os.remove(f)

        except Exception as e:
            status.error(f"エラー：{e}")
            progress_bar.empty()
