"""
各フィルタ（LPF, HPF, BPF, NF）の周波数振幅特性を表示するスクリプト

仕様:
- サンプル点数: 8000点
- LPF: 遮断周波数 2kHz
- HPF: 遮断周波数 2kHz
- BPF: 遮断周波数 2kHz-6kHz
- NF: 除去周波数 2kHz, 減衰係数 r=0.9

制限事項:
- サンプリング周波数は16kHzに固定
- フィルタ次数は100に固定（FIRフィルタの場合）
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib import rcParams


# 日本語フォントの設定
rcParams['font.family'] = 'Hiragino Sans'  # macOS用


class Constants:
    """フィルタ設計の定数クラス"""
    # フィルタパラメータ
    CUTOFF_FREQ = 2000  # 遮断周波数 [Hz]
    LOW_CUTOFF = 2000  # 下限遮断周波数 [Hz]
    HIGH_CUTOFF = 6000  # 上限遮断周波数 [Hz]
    SAMPLING_FREQ = 16000  # サンプリング周波数 [Hz]
    FILTER_ORDER = 100  # フィルタ次数
    R = 0.9  # ノッチフィルタの減衰係数


def design_lowpass_filter():
    """ローパスフィルタを設計"""
    normalized_cutoff = Constants.CUTOFF_FREQ / (Constants.SAMPLING_FREQ / 2)
    return signal.firwin(
        Constants.FILTER_ORDER + 1,
        normalized_cutoff,
        window='hamming',
        pass_zero=True
    )


def design_highpass_filter():
    """ハイパスフィルタを設計"""
    normalized_cutoff = Constants.CUTOFF_FREQ / (Constants.SAMPLING_FREQ / 2)
    return signal.firwin(
        Constants.FILTER_ORDER + 1,
        normalized_cutoff,
        window='hamming',
        pass_zero=False
    )


def design_bandpass_filter():
    """バンドパスフィルタを設計"""
    normalized_low = Constants.LOW_CUTOFF / (Constants.SAMPLING_FREQ / 2)
    normalized_high = Constants.HIGH_CUTOFF / (Constants.SAMPLING_FREQ / 2)
    return signal.firwin(
        Constants.FILTER_ORDER + 1,
        [normalized_low, normalized_high],
        window='hamming',
        pass_zero=False
    )


def design_notch_filter():
    """ノッチフィルタを設計"""
    omega = 2 * np.pi * Constants.CUTOFF_FREQ / Constants.SAMPLING_FREQ
    b = np.array([1, -2 * np.cos(omega), 1])
    a = np.array([
        1,
        -2 * Constants.R * np.cos(omega),
        Constants.R * Constants.R
    ])
    return b, a


def plot_frequency_response(w, h, title, ax):
    """
    周波数応答をプロットする関数

    Parameters:
    -----------
    w : numpy.ndarray
        周波数軸
    h : numpy.ndarray
        周波数応答
    title : str
        プロットのタイトル
    ax : matplotlib.axes.Axes
        プロットする軸
    """
    # 振幅特性をデシベルに変換
    magnitude = 20 * np.log10(np.abs(h))
    
    # プロット
    ax.plot(w * Constants.SAMPLING_FREQ / (2 * np.pi), magnitude)
    ax.set_title(title)
    ax.set_xlabel('周波数 [Hz]')
    ax.set_ylabel('振幅 [dB]')
    ax.grid(True)
    ax.set_xlim(0, Constants.SAMPLING_FREQ / 2)
    ax.set_ylim(-60, 5)


def main():
    """メイン処理"""
    # フィルタを設計
    lpf_coeffs = design_lowpass_filter()
    hpf_coeffs = design_highpass_filter()
    bpf_coeffs = design_bandpass_filter()
    notch_b, notch_a = design_notch_filter()
    
    # 周波数応答を計算
    w_lpf, h_lpf = signal.freqz(lpf_coeffs, worN=8000)
    w_hpf, h_hpf = signal.freqz(hpf_coeffs, worN=8000)
    w_bpf, h_bpf = signal.freqz(bpf_coeffs, worN=8000)
    w_notch, h_notch = signal.freqz(notch_b, notch_a, worN=8000)
    
    # プロットの設定
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('フィルタの周波数振幅特性', fontsize=16)
    
    # 各フィルタの周波数応答をプロット
    plot_frequency_response(w_lpf, h_lpf, 'ローパスフィルタ (2kHz)', ax1)
    plot_frequency_response(w_hpf, h_hpf, 'ハイパスフィルタ (2kHz)', ax2)
    plot_frequency_response(w_bpf, h_bpf, 'バンドパスフィルタ (2kHz-6kHz)', ax3)
    plot_frequency_response(w_notch, h_notch, 'ノッチフィルタ (2kHz)', ax4)
    
    # レイアウトの調整
    plt.tight_layout()
    
    # プロットを保存
    plt.savefig('filter_frequency_responses.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
