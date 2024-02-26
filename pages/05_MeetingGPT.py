import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os

has_transcript = os.path.exists("./.cache/test2.txt")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as dest_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            dest_file.write(transcript["text"])


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace(".mp4", ".mp3")
    command = [
        "ffmpeg",
        "-y",  # overwrite
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)
    return audio_path


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


st.set_page_config(page_title="MeetingGPT", page_icon="🔉")

st.markdown(
    """
# MeetingGPT
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.
            
Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "mov", "avi", "mkv"],
    )

if video:
    chunks_path = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        transcript_path = video_path.replace(".mp4", ".txt")
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        audio_path = extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_path)
        status.update(label="Trnascribing audio...")
        transcribe_chunks(chunks_path, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )
    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())
