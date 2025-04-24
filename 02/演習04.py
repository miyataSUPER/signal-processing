"""
ノッチフィルタの設計と実装

仕様:
- 除去周波数: 2kHz
- 減衰係数: r=0.9
- サンプリング周波数: 16kHz
- 入力信号: 白色雑音 + 2kHzの正弦波

制限事項:
- 出力は16bitのwav形式で保存されます

TODO:
- フィルタの周波数特性の可視化
- 入力信号と出力信号の比較表示
"""

import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォントの設定
rcParams['font.family'] = 'Hiragino Sans'  # macOS用


def design_notch_filter(notch_freq, sampling_freq, r):
    """
    ノッチフィルタを設計する関数

    Parameters:
    -----------
    notch_freq : float
        除去周波数 [Hz]
    sampling_freq : float
        サンプリング周波数 [Hz]
    r : float
        減衰係数（0 < r < 1）

    Returns:
    --------
    tuple
        (b, a) フィルタ係数
    """
    # 正規化された周波数を計算
    omega = 2 * np.pi * notch_freq / sampling_freq
    
    # フィルタ係数を計算
    b = np.array([1, -2 * np.cos(omega), 1])
    a = np.array([1, -2 * r * np.cos(omega), r * r])
    
    return b, a


def generate_input_signal(duration, sampling_freq, notch_freq):
    """
    入力信号を生成する関数（白色雑音 + 正弦波）

    Parameters:
    -----------
    duration : float
        信号の長さ [秒]
    sampling_freq : float
        サンプリング周波数 [Hz]
    notch_freq : float
        正弦波の周波数 [Hz]

    Returns:
    --------
    numpy.ndarray
        生成された信号
    """
    # 時間軸を生成
    t = np.arange(0, duration, 1/sampling_freq)
    
    # 白色雑音を生成
    noise = np.random.normal(0, 0.1, len(t))
    
    # 正弦波を生成
    sine_wave = np.sin(2 * np.pi * notch_freq * t)
    
    # 信号を合成
    signal = noise + sine_wave
    
    return signal


def apply_filter(input_signal, b, a):
    """
    フィルタを信号に適用する関数

    Parameters:
    -----------
    input_signal : numpy.ndarray
        入力信号
    b : numpy.ndarray
        分子のフィルタ係数
    a : numpy.ndarray
        分母のフィルタ係数

    Returns:
    --------
    numpy.ndarray
        フィルタ適用後の信号
    """
    return signal.lfilter(b, a, input_signal)


def plot_signals(input_signal, output_signal, sampling_freq, notch_freq):
    """
    入力信号と出力信号をプロットする関数

    Parameters:
    -----------
    input_signal : numpy.ndarray
        入力信号
    output_signal : numpy.ndarray
        出力信号
    sampling_freq : float
        サンプリング周波数 [Hz]
    notch_freq : float
        除去周波数 [Hz]
    """
    # 時間軸を生成
    t = np.arange(len(input_signal)) / sampling_freq
    
    # プロットの設定
    plt.figure(figsize=(12, 8))
    
    # 入力信号のプロット
    plt.subplot(2, 1, 1)
    plt.plot(t, input_signal)
    plt.title('入力信号（白色雑音 + 2kHz正弦波）')
    plt.xlabel('時間 [秒]')
    plt.ylabel('振幅')
    plt.grid(True)
    
    # 出力信号のプロット
    plt.subplot(2, 1, 2)
    plt.plot(t, output_signal)
    plt.title('出力信号（2kHz成分除去後）')
    plt.xlabel('時間 [秒]')
    plt.ylabel('振幅')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('notch_filter_comparison.png')
    plt.close()


def main():
    # パラメータ設定
    NOTCH_FREQ = 2000  # 除去周波数 [Hz]
    SAMPLING_FREQ = 16000  # サンプリング周波数 [Hz]
    R = 0.9  # 減衰係数
    DURATION = 1.0  # 信号の長さ [秒]
    
    try:
        # ノッチフィルタを設計
        b, a = design_notch_filter(NOTCH_FREQ, SAMPLING_FREQ, R)
        
        # 入力信号を生成
        input_signal = generate_input_signal(
            DURATION,
            SAMPLING_FREQ,
            NOTCH_FREQ
        )
        
        # フィルタを適用
        output_signal = apply_filter(input_signal, b, a)
        
        # 信号をプロット
        plot_signals(input_signal, output_signal, SAMPLING_FREQ, NOTCH_FREQ)
        
        # 信号を保存
        sf.write('input04.wav', input_signal, SAMPLING_FREQ)
        sf.write('output04.wav', output_signal, SAMPLING_FREQ)
        
        print("フィルタ処理が完了しました。")
        print("- input04.wav: 入力信号")
        print("- output04.wav: 出力信号")
        print("- notch_filter_comparison.png: 信号の比較プロット")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")


if __name__ == "__main__":
    main()
