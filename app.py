import streamlit as st
import whisper
import yt_dlp
import subprocess
import os
import uuid
import re

st.set_page_config(page_title="YouTube文字起こしツール")

@st.cache_resource(show_spinner="Whisperモデルを読み込み中…（初回のみ数十秒かかります）")
def load_model():
    return whisper.load_model("tiny")  # 軽量＆高速

# YouTube音声を.m4aでDL → wav変換
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

# 900秒ごとに分割
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

# 日本語整形
def format_text_japanese(raw_text):
    text = re.sub(r'(?<=[。！？])', '\n', raw_text)
    text = re.sub(r'([^\n]{20,40}?)(が|ので|けど|のに|そして|また|つまり)', r'\1、\2', text)
    text = re.sub(r'\n{2,}', '\n', text).strip()
    return text

# UI
st.title("🎙️ YouTube文字起こしツール（完全無料公開版）")
url = st.text_input("YouTube動画のURLを入力してください：")

if st.button("▶️ 文字起こし開始"):
    if not url:
        st.error("まず URL を入力してください")
    else:
        with st.spinner("処理中です…しばらくお待ちください"):
            try:
                model = load_model()  # モデルを先に取得
                temp_id = str(uuid.uuid4())
                wav_file, m4a_file = download_and_convert(url, temp_id)

                chunks = split_audio(wav_file)
                st.success(f"{len(chunks)} チャンクに分割されました")

                texts = []
                for i, c in enumerate(chunks):
                    texts.append(model.transcribe(c, language="ja")["text"])

                full = "\n".join(texts)
                formatted = format_text_japanese(full)
                st.subheader("📝 整形済み文字起こし")
                st.text_area("", formatted, height=400)
                st.download_button("📋 全文コピー", formatted, file_name="transcription.txt")

                # cleanup
                for f in [wav_file, m4a_file] + chunks:
                    if os.path.exists(f):
                        os.remove(f)

            except Exception as e:
                st.error(f"エラー：{e}")
