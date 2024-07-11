import os
import subprocess
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
from scipy.fftpack import fft, ifft
import speech_recognition as sr

# 必要なディレクトリの確認と作成
for dir in ['./input', './output', './sample', './setting', './artWork']:
    if not os.path.exists(dir):
        os.makedirs(dir)

def remove_silence(audio):
    # 2秒以上の無音を1秒に短縮
    chunks = split_on_silence(audio, min_silence_len=2000, silence_thresh=-40)
    processed_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        silence_chunk = AudioSegment.silent(duration=1000)
        processed_chunks.append(silence_chunk)
        processed_chunks.append(chunk)
    return sum(processed_chunks)

def remove_lip_noise_fft(audio, sample, threshold=0.5):
    audio_array = np.array(audio.get_array_of_samples())
    sample_array = np.array(sample.get_array_of_samples())
    
    audio_fft = fft(audio_array)
    sample_fft = fft(sample_array, n=len(audio_array))
    
    audio_power = np.abs(audio_fft)
    sample_power = np.abs(sample_fft)
    
    mask = audio_power > (threshold * sample_power)
    audio_fft_filtered = audio_fft * mask
    
    audio_filtered = np.real(ifft(audio_fft_filtered))
    
    return AudioSegment(audio_filtered.astype(np.int16).tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width, channels=audio.channels)

def detect_fillers(audio_segment, filler_words=["あー", "えー", "あのー"]):
    recognizer = sr.Recognizer()
    fillers = []
    
    chunks = split_on_silence(audio_segment, min_silence_len=500, silence_thresh=-40)
    
    for i, chunk in enumerate(chunks):
        chunk.export(f"temp_chunk_{i}.wav", format="wav")
        
        with sr.AudioFile(f"temp_chunk_{i}.wav") as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language="ja-JP")
                
                if any(filler in text for filler in filler_words):
                    fillers.append((chunk.duration_seconds * 1000 * i, chunk.duration_seconds * 1000 * (i + 1)))
            except sr.UnknownValueError:
                pass
        
        os.remove(f"temp_chunk_{i}.wav")  # 一時ファイルの削除
    
    return fillers

def remove_fillers(audio, fillers, reduction_factor=0.2):
    audio_array = np.array(audio.get_array_of_samples())
    
    for start, end in fillers:
        start_sample = int(start * audio.frame_rate / 1000)
        end_sample = int(end * audio.frame_rate / 1000)
        audio_array[start_sample:end_sample] *= reduction_factor
    
    return AudioSegment(audio_array.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width, channels=audio.channels)

def apply_eq(input_file, output_file):
    # EQ設定の適用
    with open('./setting/eq.txt', 'r') as f:
        eq_settings = f.read().strip().split('\n')
    
    eq_filter = ','.join([f"equalizer=f={freq}:width_type=o:width=1:g={gain}" for freq, gain in [line.split(',') for line in eq_settings]])
    
    command = [
        'ffmpeg', '-i', input_file,
        '-af', eq_filter,
        '-c:a', 'pcm_s16le',
        output_file
    ]
    
    subprocess.run(command, check=True)

def process_audio(input_file):
    print("処理を開始します...")

    # 入力ファイルの読み込み
    print("音声ファイルを読み込んでいます...")
    audio = AudioSegment.from_wav(input_file)

    # 無音部分の処理
    print("無音部分を処理しています...")
    audio = remove_silence(audio)

    # リップノイズの除去
    print("リップノイズを除去しています...")
    lip_noise_sample = AudioSegment.from_wav('./sample/lip_noise.wav')
    audio = remove_lip_noise_fft(audio, lip_noise_sample)

    # フィラー語の除去
#    print("フィラー語を検出しています...")
#    fillers = detect_fillers(audio)
#    print("フィラー語を除去しています...")
#    audio = remove_fillers(audio, fillers)

    # 一時的なWAVファイルとして保存
    temp_wav = './temp_processed.wav'
    audio.export(temp_wav, format='wav')

    # ラウドネスノーマライズとEQ処理
    print("ラウドネスノーマライズとEQ処理を適用しています...")
    temp_eq_wav = './temp_eq.wav'
    apply_eq(temp_wav, temp_eq_wav)

    # MP3への変換とメタデータの追加
    print("MP3に変換し、メタデータを追加しています...")
    output_file = './output/processed_output.mp3'
    
    # タグ情報の読み込み
    with open('./setting/tag.txt', 'r', encoding='utf-8') as f:
        tag_data = dict(line.strip().split('=') for line in f)

    # FFmpegコマンドの構築
    command = [
        'ffmpeg', '-i', temp_eq_wav,
        '-i', './artWork/artwork.jpg',
        '-filter:a', f"loudnorm=I=-16.0:LRA=11:TP=-1.5",
        '-c:a', 'libmp3lame', '-b:a', '96k',
        '-map', '0:0', '-map', '1:0',
        '-id3v2_version', '3',
        '-metadata', f"title={tag_data['タイトル']}",
        '-metadata', f"album={tag_data['アルバム']}",
        '-metadata', f"year={tag_data['年']}",
        '-metadata', f"genre={tag_data['ジャンル']}",
        '-metadata', f"artist={tag_data['参加アーティスト']}",
        '-metadata', f"track={tag_data['トラック番号']}",
        output_file
    ]

    subprocess.run(command, check=True)

    # 一時ファイルの削除
    os.remove(temp_wav)
    os.remove(temp_eq_wav)

    print("処理が完了しました。出力ファイル:", output_file)

# メイン処理
input_files = [f for f in os.listdir('./input') if f.endswith('.wav')]
for input_file in input_files:
    process_audio(os.path.join('./input', input_file))