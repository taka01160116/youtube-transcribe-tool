import streamlit as st
import whisper
import yt_dlp
import subprocess
import os
import uuid
import re
import time

st.set_page_config(page_title="YouTube文字起こしツール")

@st.cache_resource(show_spinner="Whisperモデルを読み込み中…（初回のみ数十秒）")
def load_model():
    return whisper.load_model("tiny")

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

def split_audio(input_file, chunk_length=900):
    chunks = []
    idx = 0
    while True:
        out = f"{input_file}_part{idx}.wav"
        cmd = ["ffmpeg", "-y", "-i", input_file, "-ss", str(idx * chunk_length),
               "-t", str(chunk_length), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not os.path.exists(out):
            break
        chunks.append(out)
        idx += 1
    return chunks

def format_text_japanese(raw_text):
    text = re.sub(r'(?<=[。！？])', '\n', raw_text)
    text = re.sub(r'([^\n]{20,40}?)(が|ので|けど|のに|そして|また|つまり)', r'\1、\2', text)
    text = re.sub(r'\n{2,}', '\n', text).strip()
    return text

# --- UI ---
st.title("🎙️ YouTube文字起こしツール（完全無料公開版）")
url = st.text_input("YouTube動画のURLを入力してください：")

if st.button("▶️ 文字起こし開始"):
    if not url:
        st.error("まず URL を入力してください")
    else:
        status = st.empty()
        progress_bar = st.progress(0, text="開始準備中…")
        try:
            model = load_model()

            temp_id = str(uuid.uuid4())
            status.info("🔄 音声ダウンロード中…")
            wav_file, m4a_file = download_and_convert(url, temp_id)

            status.info("🔄 音声分割中…")
            chunks = split_audio(wav_file)
            status.success(f"✅ {len(chunks)} チャンクに分割されました")

            texts = []
            durations = []
            total_chunks = len(chunks)

            for i, c in enumerate(chunks):
                start = time.time()
                status_text = f"🧠 {i+1}/{total_chunks} チャンク文字起こし中…"

                if durations:
                    avg = sum(durations) / len(durations)
                    remaining_sec = int(avg * (total_chunks - i))
                    minutes = remaining_sec // 60
                    seconds = remaining_sec % 60
                    status_text += f"（残り：約 {minutes}分 {seconds}秒）"

                percent_complete = int(((i) / total_chunks) * 100)
                progress_bar.progress(percent_complete, text=status_text)

                result = model.transcribe(c, language="ja")["text"]
                texts.append(result)
                durations.append(time.time() - start)

            progress_bar.progress(100, text="🎉 文字起こし完了！")

            full = "\n".join(texts)
            formatted = format_text_japanese(full)

            st.subheader("📝 整形済み文字起こし")
            st.text_area("", formatted, height=400)
            st.download_button("📋 全文コピー", formatted, file_name="transcription.txt")

            for f in [wav_file, m4a_file] + chunks:
                if os.path.exists(f):
                    os.remove(f)

            status.success("✅ 全工程が完了しました")

        except Exception as e:
            status.error(f"エラー：{e}")
            progress_bar.empty()
