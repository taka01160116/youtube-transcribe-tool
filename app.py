import streamlit as st
import whisper
import yt_dlp
import subprocess
import os
import uuid
import re

st.set_page_config(page_title="YouTube文字起こしツール")

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

st.title("🎙️ YouTube文字起こしツール（完全無料公開版）")
url = st.text_input("YouTube動画のURLを入力してください：")

if st.button("▶️ 文字起こし開始"):
    if not url:
        st.error("まず URL を入力してください")
    else:
        with st.spinner("処理中です…しばらくお待ちください"):
            try:
                temp_id = str(uuid.uuid4())
                wav_file = f"{temp_id}.wav"

                # 音声ダウンロード
                download_audio_ytdlp(url, wav_file)
                if not os.path.exists(wav_file):
                    raise FileNotFoundError(f"{wav_file} が作成されませんでした")

                # 音声分割と文字起こし
                chunks = split_audio(wav_file)
                model = whisper.load_model("base")
                full = ""
                for i, c in enumerate(chunks):
                    st.info(f"{i+1}/{len(chunks)} チャンク処理中…")
                    full += model.transcribe(c, language="ja")["text"] + "\n"

                # 整形・表示
                formatted = format_text_japanese(full)
                st.subheader("📝 整形済み文字起こし")
                st.text_area("", formatted, height=400)
                st.download_button("📋 全文コピー", formatted, file_name="transcription.txt")

                # cleanup
                if os.path.exists(wav_file): os.remove(wav_file)
                for c in chunks:
                    if os.path.exists(c): os.remove(c)

            except Exception as e:
                st.error(f"エラー：{e}")
