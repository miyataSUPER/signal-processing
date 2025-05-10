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
import matplotlib as mpl

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'  # macOS用
# plt.rcParams['font.family'] = 'MS Gothic'  # Windows用
# plt.rcParams['font.family'] = 'Noto Sans CJK JP'  # Linux用

# フォントが見つからない場合のフォールバック設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'MS Gothic', 'Noto Sans CJK JP', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止


def plot_amplitude_spectrum(f, t, Zxx, fs=16000):
    """
    振幅スペクトルを描画する関数

    Parameters:
    -----------
    f : numpy.ndarray
        周波数軸
    t : numpy.ndarray
        時間軸
    Zxx : numpy.ndarray
        STFTの結果
    fs : int, optional
        サンプリング周波数（デフォルト: 16000Hz）
    """
    # 振幅スペクトルの計算（dBスケール）
    amplitude_spectrum = 20 * np.log10(np.abs(Zxx) + 1e-10)
    
    # プロットの設定
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, amplitude_spectrum, shading='gouraud')
    plt.colorbar(label='振幅 [dB]')
    plt.title('振幅スペクトル')
    plt.xlabel('時間 [秒]')
    plt.ylabel('周波数 [Hz]')
    plt.ylim(0, fs/2)  # ナイキスト周波数まで表示
    plt.tight_layout()
    plt.savefig('amplitude_spectrum.png')
    plt.close()

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
        (STFT結果, フレーム長, シフト長, 周波数軸, 時間軸)
    """
    # フレーム長をサンプル数に変換
    frame_length = int(fs * frame_length_ms / 1000)
    # シフト長を計算
    shift_length = int(frame_length * (1 - overlap))
    
    # ハニング窓の作成
    window = np.hanning(frame_length)
    
    # STFTの実行
    f, t, Zxx = signal.stft(signal_data, fs=fs, window=window,
                           nperseg=frame_length,
                           noverlap=frame_length-shift_length)
    
    return Zxx, frame_length, shift_length, f, t

def istft_synthesis(stft_result, fs=16000, frame_length=None,
                   shift_length=None):
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
        input_file = "/Users/miyatahiroshi/homework/signal-processing/04/01_testsound_r1.wav"
        signal_data, fs = sf.read(input_file)
        
        if fs != 16000:
            raise ValueError(f"サンプリング周波数が16kHzではありません: {fs}Hz")
        
        # STFT解析
        stft_result, frame_length, shift_length, f, t = stft_analysis(signal_data)
        
        # 振幅スペクトルの描画
        plot_amplitude_spectrum(f, t, stft_result)
        
        # ISTFT再構築
        reconstructed_signal = istft_synthesis(stft_result,
                                             frame_length=frame_length,
                                             shift_length=shift_length)
        
        # 結果の保存
        output_file = "reconstructed.wav"
        sf.write(output_file, reconstructed_signal, fs)
        
        print(f"処理が完了しました。")
        print(f"- 再構築された音声は {output_file} に保存されました。")
        print(f"- 振幅スペクトルは amplitude_spectrum.png に保存されました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
