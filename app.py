import streamlit as st
import whisper
import yt_dlp
import subprocess
import os
import uuid
import re
import time
import torch
import multiprocessing
from glob import glob

st.set_page_config(page_title="YouTube文字起こしツール")

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

# Whisperモデルをキャッシュ
@st.cache_resource(show_spinner="Whisperモデルを読み込み中…")
def load_model(model_size):
    model = whisper.load_model(model_size)
    return model.to(DEVICE)

# 無音チェック
def is_silent_audio(file_path, threshold_db=-40):
    result = subprocess.run(
        ["ffmpeg", "-i", file_path, "-af", "volumedetect", "-f", "null", "-"],
        stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    stderr = result.stderr
    match = re.search(r"mean_volume: (-?\d+\.?\d*) dB", stderr)
    if match:
        return float(match.group(1)) < threshold_db
    return True

# YouTube音声ダウンロード＋WAV変換
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

# 音声分割
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

# 整形
def format_text_japanese(raw_text):
    text = re.sub(r'(?<=[。！？])', '\n', raw_text)
    text = re.sub(r'([^\n]{20,40}?)(が|ので|けど|のに|そして|また|つまり)', r'\1、\2', text)
    text = re.sub(r'\n{2,}', '\n', text).strip()
    return text

# Whisper推論（multiprocessingでタイムアウト付き）
def transcribe_chunk_with_timeout(model, chunk, timeout=30):
    def worker(pipe_conn, model_path, chunk_path):
        try:
            model_local = whisper.load_model(model_path).to(DEVICE)
            result = model_local.transcribe(chunk_path, language="ja", fp16=False, no_speech_threshold=0.1)["text"]
            pipe_conn.send(result)
        except Exception as e:
            pipe_conn.send(f"ERROR: {e}")

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=worker, args=(child_conn, model_size, chunk))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return "ERROR: タイムアウトしました"
    return parent_conn.recv()

# --- UI ---
st.title("🎙️ YouTube文字起こしツール（完全無料公開版）")

url = st.text_input("🎥 YouTube動画のURLを入力してください：")
model_size = st.selectbox("⚙️ 使用するWhisperモデルを選択：", ["tiny", "base", "medium"], index=1)

st.subheader("📝 整形済み文字起こし")
output_placeholder = st.empty()
copy_btn_placeholder = st.empty()
formatted_text = ""

if st.button("▶️ 文字起こし開始"):
    if not url:
        st.error("まず URL を入力してください")
    else:
        status = st.empty()
        progress_bar = st.progress(0, text="開始準備中…")

        try:
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

                start = time.time()
                status_text = f"🧠 {i+1}/{total_chunks} チャンク文字起こし中…"
                if durations:
                    avg = sum(durations) / len(durations)
                    remaining = int(avg * (total_chunks - i))
                    mins, secs = divmod(remaining, 60)
                    status_text += f"（残り：約 {mins}分 {secs}秒）"
                progress_bar.progress(min((i+1) / total_chunks, 1.0), text=status_text)

                st.info(f"🧠 {chunk} の文字起こしを開始します…")
                print(f"--- TRANSCRIBE開始: {chunk} ---")
                result = transcribe_chunk_with_timeout(model_size, chunk, timeout=60)
                print(f"--- TRANSCRIBE終了: {chunk} ---")

                if result.startswith("ERROR"):
                    st.error(f"❌ {chunk} の文字起こしに失敗: {result}")
                    continue

                texts.append(result)
                durations.append(time.time() - start)
                st.success(f"✅ {chunk} 完了")

            progress_bar.progress(1.0, text="🎉 文字起こし完了！")

            full = "\n".join(texts)
            formatted_text = format_text_japanese(full)

            if not formatted_text.strip():
                status.warning("⚠️ 有効な音声が検出されませんでした。")
            else:
                status.success("✅ 全工程が完了しました")
                output_placeholder.text_area("以下が文字起こしの全文です：", formatted_text, height=400)
                copy_btn_placeholder.download_button("📋 全文コピー（テキストファイル）", formatted_text, file_name="transcription.txt")

            for f in [wav_file, m4a_file] + chunks:
                if os.path.exists(f):
                    os.remove(f)

        except Exception as e:
            status.error(f"エラー：{e}")
            progress_bar.empty()
