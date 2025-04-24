"""
正弦波に対するFFTを実行し、結果を確認するプログラム

仕様:
- サンプリング周波数: 1000Hz
- サンプル点数: 1000点（N点）
- 正弦波の周波数: 100Hz
- 振り幅: 1.0

表示内容:
- 時間領域の正弦波信号
- 振り幅スペクトル（絶対値）
- 位相スペクトル（angle）

注意:
- 時間領域のプロットは最初の0.1秒間（N/10点）のみ表示
- スペクトルは正の周波数成分（N/2点）のみ表示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# 日本語フォントの設定
rcParams['font.family'] = 'Hiragino Sans'  # macOS用


def main():
    # パラメータ設定
    fs = 1000  # サンプリング周波数 [Hz]
    N = 1000   # サンプル数
    f0 = 100   # 正弦波の周波数 [Hz]
    t = np.arange(N) / fs  # 時間軸 [s]
    
    # 正弦波信号の生成
    sine_wave = np.sin(2 * np.pi * f0 * t)
    
    # FFTの実行
    fft_result = np.fft.fft(sine_wave)
    freq = np.fft.fftfreq(N, 1/fs)  # 周波数軸 [Hz]
    
    # 振り幅スペクトルと位相スペクトルの計算
    amplitude = np.abs(fft_result) * 2 / N  # 振り幅の正規化
    phase = np.angle(fft_result)
    
    # 結果の可視化
    plt.figure(figsize=(12, 10))
    
    # 時間領域のプロット
    plt.subplot(3, 1, 1)
    plt.plot(t[:N//10], sine_wave[:N//10])  # 最初の0.1秒間のみ表示
    plt.title('時間領域の正弦波信号（最初の0.1秒）')
    plt.xlabel('時間 [s]')
    plt.ylabel('振り幅')
    plt.grid(True)
    
    # 振り幅スペクトルのプロット（正の周波数のみ）
    plt.subplot(3, 1, 2)
    plt.plot(freq[:N//2], amplitude[:N//2])
    plt.title('振り幅スペクトル')
    plt.xlabel('周波数 [Hz]')
    plt.ylabel('振り幅')
    plt.grid(True)
    
    # 位相スペクトルのプロット（正の周波数のみ）
    plt.subplot(3, 1, 3)
    plt.plot(freq[:N//2], phase[:N//2])
    plt.title('位相スペクトル')
    plt.xlabel('周波数 [Hz]')
    plt.ylabel('位相 [rad]')
    plt.grid(True)
    
    plt.tight_layout()
    
    # PNGファイルとして保存
    plt.savefig('03/sine_wave_fft_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 結果の確認（100Hz付近のピーク）
    peak_index = np.argmax(amplitude[:N//2])
    peak_freq = freq[peak_index]
    peak_amplitude = amplitude[peak_index]
    peak_phase = phase[peak_index]
    
    print(f'検出された周波数成分:')
    print(f'周波数: {peak_freq:.1f} Hz')
    print(f'振り幅: {peak_amplitude:.6f}')
    print(f'位相: {peak_phase:.6f} rad')


if __name__ == "__main__":
    main()
