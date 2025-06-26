import streamlit as st
import whisper
import yt_dlp
import subprocess
import os
import uuid
import re
import time

st.set_page_config(page_title="YouTube文字起こしツール")

# モデル読み込み（初回のみ時間がかかる）
@st.cache_resource(show_spinner="Whisperモデルを読み込み中…（初回のみ数十秒）")
def load_model():
    return whisper.load_model("tiny")  # 軽量モデル使用

# YouTubeから音声DL＆WAV変換
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

# 音声を900秒（15分）単位で分割
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

# 日本語整形（句点や接続詞で改行）
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
        status = st.empty()  # 状態表示用プレースホルダ
        try:
            # モデル読み込み
            model = load_model()

            # 音声ダウンロードと変換
            temp_id = str(uuid.uuid4())
            status.info("🔄 音声ダウンロード中…")
            wav_file, m4a_file = download_and_convert(url, temp_id)

            # 分割
            status.info("🔄 音声分割中…")
            chunks = split_audio(wav_file)
            status.success(f"✅ {len(chunks)} チャンクに分割されました")

            # 文字起こしと時間計測
            texts = []
            durations = []
            for i, c in enumerate(chunks):
                start = time.time()
                progress_text = f"🧠 {i+1}/{len(chunks)} チャンク文字起こし中…"

                # 残り予測時間の表示
                if durations:
                    avg = sum(durations) / len(durations)
                    remaining = int(avg * (len(chunks) - i))
                    progress_text += f"（残り：約 {remaining} 秒）"

                status.info(progress_text)

                result = model.transcribe(c, language="ja")["text"]
                texts.append(result)
                durations.append(time.time() - start)

            # 整形して出力
            full = "\n".join(texts)
            formatted = format_text_japanese(full)

            st.subheader("📝 整形済み文字起こし")
            st.text_area("", formatted, height=400)
            st.download_button("📋 全文コピー", formatted, file_name="transcription.txt")

            # クリーンアップ
            for f in [wav_file, m4a_file] + chunks:
                if os.path.exists(f):
                    os.remove(f)

            status.success("🎉 文字起こし完了！")

        except Exception as e:
            status.error(f"エラー：{e}")
