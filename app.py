# 完全修正版：YouTube文字起こしツール（Streamlit Cloud対応）
import streamlit as st
import whisper
import yt_dlp
import os
import uuid
import subprocess
import re

st.set_page_config(page_title="YouTube文字起こしツール")

# 音声ファイルのダウンロード関数（yt_dlp使用）
def download_audio_ytdlp(url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# 音声ファイルを分割する関数（900秒ごと）
def split_audio(input_file, chunk_length=900):
    chunks = []
    idx = 0
    while True:
        out = f"{input_file}_part{idx}.wav"
        cmd = ["ffmpeg", "-y", "-i", input_file, "-ss", str(idx * chunk_length),
               "-t", str(chunk_length), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not os.path.exists(out):
            break
        chunks.append(out)
        idx += 1
    return chunks

# 日本語向け整形関数
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
                temp = str(uuid.uuid4())
                audio_path = f"{temp}.wav"
                download_audio_ytdlp(url, audio_path)

                chunks = split_audio(audio_path)
                model = whisper.load_model("base")
                full_text = ""
                for i, c in enumerate(chunks):
                    st.info(f"{i+1}/{len(chunks)} チャンク処理中…")
                    result = model.transcribe(c, language="ja")
                    full_text += result["text"] + "\n"

                formatted = format_text_japanese(full_text)
                st.subheader("📝 整形済み文字起こし")
                st.text_area("", formatted, height=400)
                st.download_button("📋 全文コピー", formatted, file_name="transcription.txt")

                # 後片付け
                os.remove(audio_path)
                for c in chunks:
                    os.remove(c)

            except Exception as e:
                st.error(f"エラー：{e}")
