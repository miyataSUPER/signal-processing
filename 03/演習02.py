"""
インパルス信号に対するFFTを実行し、結果を確認するプログラム

仕様:
- サンプリング周波数: 1000Hz
- サンプル点数: 1000点（N点）
- インパルス位置: 中央（N/2点目 = 500点目）
- 振り幅: 1.0

表示内容:
- 時間領域のインパルス信号
- 振り幅スペクトル（絶対値）
- 位相スペクトル（angle）
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
    t = np.arange(N) / fs  # 時間軸 [s]
    
    # インパルス信号の生成
    impulse = np.zeros(N)
    impulse[N//2] = 1.0  # 中央にインパルスを配置
    
    # FFTの実行
    fft_result = np.fft.fft(impulse)
    freq = np.fft.fftfreq(N, 1/fs)  # 周波数軸 [Hz]
    
    # 振り幅スペクトルと位相スペクトルの計算
    amplitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    
    # 結果の可視化
    plt.figure(figsize=(12, 10))
    
    # 時間領域のプロット
    plt.subplot(3, 1, 1)
    plt.plot(t, impulse)
    plt.title('時間領域のインパルス信号')
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
    plt.savefig('03/impulse_fft_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main() 
