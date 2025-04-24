"""
正弦波に窓関数を適用してFFTを実行し、結果を確認するプログラム

仕様:
- サンプリング周波数: 1000Hz
- サンプル点数: 1000点（N点）
- 正弦波の周波数: 100Hz
- 振り幅: 1.0

窓関数:
- 矩形窓（デフォルト）
- ハミング窓
- ハニング窓

表示内容:
- 時間領域の信号（窓関数適用前後）
- 振り幅スペクトル（各窓関数での比較）
- 位相スペクトル（各窓関数での比較）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# 日本語フォントの設定
rcParams['font.family'] = 'Hiragino Sans'  # macOS用


def apply_window(signal, window_type):
    """
    信号に窓関数を適用する

    Parameters:
    -----------
    signal : numpy.ndarray
        入力信号
    window_type : str
        窓関数の種類 ('rectangular', 'hamming', 'hanning')
    
    Returns:
    --------
    numpy.ndarray
        窓関数が適用された信号
    numpy.ndarray
        窓関数自体
    """
    N = len(signal)
    
    if window_type == 'rectangular':
        window = np.ones(N)
    elif window_type == 'hamming':
        window = np.hamming(N)
    elif window_type == 'hanning':
        window = np.hanning(N)
    else:
        raise ValueError(f'未対応の窓関数: {window_type}')
    
    return signal * window, window


def analyze_spectrum(signal, fs):
    """
    信号のスペクトル解析を行う

    Parameters:
    -----------
    signal : numpy.ndarray
        入力信号
    fs : float
        サンプリング周波数

    Returns:
    --------
    tuple
        (周波数軸, 振り幅スペクトル, 位相スペクトル)
    """
    N = len(signal)
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(N, 1/fs)
    
    # 振り幅スペクトルの正規化
    amplitude = np.abs(fft_result) * 2 / N
    phase = np.angle(fft_result)
    
    return freq, amplitude, phase


def main():
    # パラメータ設定
    fs = 1000  # サンプリング周波数 [Hz]
    N = 1000   # サンプル数
    f0 = 100   # 正弦波の周波数 [Hz]
    t = np.arange(N) / fs  # 時間軸 [s]
    
    # 正弦波信号の生成
    sine_wave = np.sin(2 * np.pi * f0 * t)
    
    # 各窓関数を適用
    windows = ['rectangular', 'hamming', 'hanning']
    colors = ['b', 'r', 'g']
    
    # プロット設定
    fig = plt.figure(figsize=(15, 12))
    
    # 1. 時間領域の信号と窓関数
    plt.subplot(3, 1, 1)
    plt.plot(t[:N//10], sine_wave[:N//10], 'k--', label='元の信号', alpha=0.5)
    
    for window_type, color in zip(windows, colors):
        windowed_signal, window = apply_window(sine_wave, window_type)
        plt.plot(t[:N//10], windowed_signal[:N//10], color, label=f'{window_type}窓適用後')
    
    plt.title('時間領域の信号（最初の0.1秒）')
    plt.xlabel('時間 [s]')
    plt.ylabel('振り幅')
    plt.grid(True)
    plt.legend()
    
    # 2. 振り幅スペクトル
    plt.subplot(3, 1, 2)
    
    for window_type, color in zip(windows, colors):
        windowed_signal, _ = apply_window(sine_wave, window_type)
        freq, amplitude, _ = analyze_spectrum(windowed_signal, fs)
        plt.plot(freq[:N//2], amplitude[:N//2], color, label=f'{window_type}窓')
    
    plt.title('振り幅スペクトル')
    plt.xlabel('周波数 [Hz]')
    plt.ylabel('振り幅')
    plt.grid(True)
    plt.legend()
    
    # 3. 位相スペクトル
    plt.subplot(3, 1, 3)
    
    for window_type, color in zip(windows, colors):
        windowed_signal, _ = apply_window(sine_wave, window_type)
        freq, _, phase = analyze_spectrum(windowed_signal, fs)
        plt.plot(freq[:N//2], phase[:N//2], color, label=f'{window_type}窓')
    
    plt.title('位相スペクトル')
    plt.xlabel('周波数 [Hz]')
    plt.ylabel('位相 [rad]')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # PNGファイルとして保存
    plt.savefig('03/windowed_sine_fft_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 各窓関数でのピーク検出結果を表示
    print('各窓関数での検出結果:')
    for window_type in windows:
        windowed_signal, _ = apply_window(sine_wave, window_type)
        freq, amplitude, phase = analyze_spectrum(windowed_signal, fs)
        
        peak_index = np.argmax(amplitude[:N//2])
        peak_freq = freq[peak_index]
        peak_amplitude = amplitude[peak_index]
        peak_phase = phase[peak_index]
        
        print(f'\n{window_type}窓:')
        print(f'  周波数: {peak_freq:.1f} Hz')
        print(f'  振り幅: {peak_amplitude:.6f}')
        print(f'  位相: {peak_phase:.6f} rad')


if __name__ == "__main__":
    main() 
