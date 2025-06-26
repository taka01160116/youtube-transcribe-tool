import streamlit as st
import whisper
from pytube import YouTube
import subprocess, os, uuid, re

st.set_page_config(page_title="YouTube文字起こしツール")

def split_audio(input_file, chunk_length=900):
    chunks = []
    idx = 0
    while True:
        out = f"{input_file}_part{idx}.wav"
        cmd = ["ffmpeg","-y","-i",input_file,"-ss",str(idx*chunk_length),
               "-t",str(chunk_length),"-acodec","pcm_s16le","-ar","16000","-ac","1", out]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not os.path.exists(out): break
        chunks.append(out); idx+=1
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
                temp = str(uuid.uuid4())
                mp4 = f"{temp}.mp4"; wav = f"{temp}.wav"
                yt = YouTube(url); yt.streams.filter(only_audio=True).first().download(filename=mp4)
                os.system(f"ffmpeg -y -i {mp4} -ar 16000 -ac 1 {wav}")
                chunks = split_audio(wav)
                model = whisper.load_model("base")
                full = ""
                for i, c in enumerate(chunks):
                    st.info(f"{i+1}/{len(chunks)} チャンク処理中…")
                    full += model.transcribe(c, language="ja")["text"] + "\n"
                formatted = format_text_japanese(full)
                st.subheader("📝 整形済み文字起こし")
                st.text_area("", formatted, height=400)
                st.download_button("📋 全文コピー", formatted, file_name="transcription.txt")
                # cleanup
                os.remove(mp4); os.remove(wav)
                for c in chunks: os.remove(c)
            except Exception as e:
                st.error(f"エラー：{e}")
