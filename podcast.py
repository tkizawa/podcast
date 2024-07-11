import os
import subprocess
import json
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

# 必要なディレクトリの確認と作成
for dir in ['./input', './output', './sample', './setting', './artWork']:
    if not os.path.exists(dir):
        os.makedirs(dir)


# EQの設定ファイルを作成
eq_settings = """32,-12.0
64,-12.0
128,-4.4
250,-1.5
500,0
1000,4.0
2000,4.0
4000,0
8000,-1.5
16000,-6.5"""

with open('./setting/eq.txt', 'w') as f:
    f.write(eq_settings)

# タグ情報のファイルを作成
tag_info = """タイトル=第737回 Windows 11がクリーンインストールできない・Microsoft 365のコース切り替え・Windows 11 6月の機能アップデート (2024/7/6)
アルバム=WoodStreamのデジタル生活(マイクロソフト系Podcast)
年=2024
ジャンル=Podcast
参加アーティスト=木澤朋和
トラック番号=1"""

with open('./setting/tag.txt', 'w', encoding='utf-8') as f:
    f.write(tag_info)

def remove_silence(audio):
    # 3秒以上の無音を1秒に短縮
    chunks = split_on_silence(audio, min_silence_len=2000, silence_thresh=-40)
    processed_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        silence_chunk = AudioSegment.silent(duration=1000)
        processed_chunks.append(silence_chunk)
        processed_chunks.append(chunk)
    return sum(processed_chunks)

def remove_lip_noise(audio, sample):
    # リップノイズの除去
    audio_array = np.array(audio.get_array_of_samples())
    sample_array = np.array(sample.get_array_of_samples())
    corr = correlate(audio_array, sample_array, mode='valid')
    threshold = 0.8 * np.max(corr)
    peaks = np.where(corr > threshold)[0]
    
    for peak in peaks:
        start = peak
        end = peak + len(sample_array)
        audio_array[start:end] = 0
    
    return AudioSegment(audio_array.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width, channels=audio.channels)

def remove_filler_words(audio, samples):
    # フィラー語の除去
    for sample in samples:
        audio = remove_lip_noise(audio, sample)
    return audio

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
#    print("リップノイズを除去しています...")
#    lip_noise_sample = AudioSegment.from_wav('./sample/lip_noise.wav')
#    audio = remove_lip_noise(audio, lip_noise_sample)

    # フィラー語の除去
    print("フィラー語を除去しています...")
    filler_samples = [AudioSegment.from_wav(os.path.join('./sample', f)) for f in os.listdir('./sample') if f.startswith('filler_')]
    audio = remove_filler_words(audio, filler_samples)

    # 一時的なWAVファイルとして保存
    temp_wav = './temp_processed.wav'
    audio.export(temp_wav, format='wav')

    # ラウドネスノーマライズとEQ処理
    print("ラウドネスノーマライズとEQ処理を適用しています...")
    temp_eq_wav = './temp_eq.wav'
#    temp_eq_wav = temp_wav
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
#    os.remove(temp_eq_wav)

    print("処理が完了しました。出力ファイル:", output_file)

# メイン処理
input_files = [f for f in os.listdir('./input') if f.endswith('.wav')]
for input_file in input_files:
    process_audio(os.path.join('./input', input_file))
