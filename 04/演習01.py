"""
音声信号のSTFT（Short-Time Fourier Transform）とISTFT（Inverse Short-Time Fourier Transform）による解析と再構築

仕様:
- サンプリング周波数: 16kHz
- フレーム長: 32ms
- 窓関数: ハニング窓
- オーバーラップ: 50%（ハーフオーバーラップ）

制限事項:
- 入力音声は16kHzのサンプリング周波数である必要があります
- 音声データは1次元のnumpy配列である必要があります

TODO:
- エラー処理の強化
- 可視化機能の追加
- パフォーマンスの最適化
"""

import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt

def stft_analysis(signal_data, fs=16000, frame_length_ms=32, overlap=0.5):
    """
    STFTによる音声信号の解析を行う関数

    Parameters:
    -----------
    signal_data : numpy.ndarray
        入力音声信号（1次元配列）
    fs : int, optional
        サンプリング周波数（デフォルト: 16000Hz）
    frame_length_ms : int, optional
        フレーム長（ミリ秒）（デフォルト: 32ms）
    overlap : float, optional
        オーバーラップ率（デフォルト: 0.5）

    Returns:
    --------
    tuple
        (STFT結果, フレーム長, シフト長)
    """
    # フレーム長をサンプル数に変換
    frame_length = int(fs * frame_length_ms / 1000)
    # シフト長を計算
    shift_length = int(frame_length * (1 - overlap))
    
    # ハニング窓の作成
    window = np.hanning(frame_length)
    
    # STFTの実行
    f, t, Zxx = signal.stft(signal_data, fs=fs, window=window,
                           nperseg=frame_length, noverlap=frame_length-shift_length)
    
    return Zxx, frame_length, shift_length

def istft_synthesis(stft_result, fs=16000, frame_length=None, shift_length=None):
    """
    ISTFTによる音声信号の再構築を行う関数

    Parameters:
    -----------
    stft_result : numpy.ndarray
        STFTの結果
    fs : int, optional
        サンプリング周波数（デフォルト: 16000Hz）
    frame_length : int, optional
        フレーム長（サンプル数）
    shift_length : int, optional
        シフト長（サンプル数）

    Returns:
    --------
    numpy.ndarray
        再構築された音声信号
    """
    # ISTFTの実行
    _, reconstructed_signal = signal.istft(stft_result, fs=fs,
                                         window='hann',
                                         nperseg=frame_length,
                                         noverlap=frame_length-shift_length)
    
    return reconstructed_signal

def main():
    """
    メイン関数：音声ファイルの読み込み、STFT解析、ISTFT再構築を実行
    """
    try:
        # 音声ファイルの読み込み（16kHzのサンプリング周波数）
        input_file = "/Users/miyatahiroshi/homework/signal-processing/04/01_testsound_r1.wav"  # 入力ファイル名
        signal_data, fs = sf.read(input_file)
        
        if fs != 16000:
            raise ValueError(f"サンプリング周波数が16kHzではありません: {fs}Hz")
        
        # STFT解析
        stft_result, frame_length, shift_length = stft_analysis(signal_data)
        
        # ISTFT再構築
        reconstructed_signal = istft_synthesis(stft_result, frame_length=frame_length,
                                             shift_length=shift_length)
        
        # 結果の保存
        output_file = "reconstructed.wav"
        sf.write(output_file, reconstructed_signal, fs)
        
        print(f"処理が完了しました。再構築された音声は {output_file} に保存されました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
