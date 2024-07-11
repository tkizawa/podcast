import os
import subprocess
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
from scipy.fftpack import fft, ifft
import glob

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

def process_audio(input_file, output_file):
    print("音声処理を開始します。")
    
    # ステップ1: 入力ファイルの読み込みと無音部分の処理
    print("ステップ 1/9: 入力ファイルの読み込みと無音部分の処理")
    audio = AudioSegment.from_wav(input_file)
    audio = remove_silence(audio)
    
    # ステップ2: リップノイズの除去
    print("ステップ 2/9: リップノイズの除去")
    lip_noise_samples = glob.glob("./sample/lip_noise*.wav")
    for sample_file in lip_noise_samples:
        lip_noise_sample = AudioSegment.from_wav(sample_file)
        audio = remove_lip_noise_fft(audio, lip_noise_sample)
    
    # ステップ3: 「あー」「えー」の除去
    print("ステップ 3/9: フィラー音の除去")
    filler_samples = glob.glob("./sample/filler*.wav")
    for sample_file in filler_samples:
        filler_sample = AudioSegment.from_wav(sample_file)
        audio = remove_lip_noise_fft(audio, filler_sample)  # 同じ方法でフィラー音も除去
    
    # 一時的なWAVファイルとして保存
    temp_wav = 'temp_processed.wav'
    audio.export(temp_wav, format='wav')
    
    # ステップ4: ラウドネスノーマライズとステップ5: EQ処理
    print("ステップ 4/9: ラウドネスノーマライズ")
    print("ステップ 5/9: グラフィックイコライザー処理")
    temp_eq_wav = 'temp_eq.wav'
    apply_eq(temp_wav, temp_eq_wav)
    
    # ステップ6: MP3形式への変換、ステップ7: タグの埋め込み、ステップ8: アートワークの埋め込み
    print("ステップ 6/9: MP3形式への変換")
    print("ステップ 7/9: タグの埋め込み")
    print("ステップ 8/9: アートワークの埋め込み")
    
    # タグ情報の読み込み
    with open('./setting/tag.txt', 'r', encoding='utf-8') as f:
        tag_data = dict(line.strip().split('=') for line in f)
    
    # アートワークファイルの取得
    artwork_file = glob.glob("./artWork/*.*")[0]
    
    # FFmpegコマンドの構築
    command = [
        'ffmpeg', '-i', temp_eq_wav,
        '-i', artwork_file,
        '-filter:a', f"loudnorm=I=-16.0:LRA=11:TP=-1.5",
        '-c:a', 'libmp3lame', '-b:a', '96k',
        '-map', '0:0', '-map', '1:0',
        '-id3v2_version', '3'
    ]
    
    # タグの追加
    for key, value in tag_data.items():
        command.extend(['-metadata', f"{key}={value}"])
    
    command.append(output_file)
    
    subprocess.run(command, check=True)
    
    # ステップ9: 一時ファイルの削除
    print("ステップ 9/9: 一時ファイルの削除")
    os.remove(temp_wav)
    os.remove(temp_eq_wav)
    
    print("音声処理が完了しました。")

def main():
    input_files = glob.glob("./input/*.wav")
    for i, input_file in enumerate(input_files, 1):
        print(f"ファイル {i}/{len(input_files)} を処理中: {input_file}")
        output_file = os.path.join("./output", os.path.splitext(os.path.basename(input_file))[0] + ".mp3")
        process_audio(input_file, output_file)

if __name__ == "__main__":
    main()