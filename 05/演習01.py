"""
音声信号のケプストラム分析によるスペクトル包絡の抽出とフォルマントの確認

仕様:
- サンプリング周波数: 16kHz
- フレーム長: 32ms
- 窓関数: ハニング窓
- オーバーラップ: 50%（ハーフオーバーラップ）
- ケプストラム次数: 20（スペクトル包絡抽出用）

制限事項:
- 入力音声は16kHzのサンプリング周波数である必要があります
- 音声データは1次元のnumpy配列である必要があります
- 入力信号の長さは窓長（32ms）以上である必要があります

TODO:
- エラー処理の強化
- フォルマント検出の精度向上
- パフォーマンスの最適化
"""

import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib as mpl

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'  # macOS用


# フォントが見つからない場合のフォールバック設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'MS Gothic', 'Noto Sans CJK JP', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止

def resample_audio(signal_data, original_fs, target_fs=16000):
    """
    音声信号をリサンプリングする関数

    Parameters:
    -----------
    signal_data : numpy.ndarray
        入力音声信号
    original_fs : int
        元のサンプリング周波数
    target_fs : int, optional
        目標のサンプリング周波数（デフォルト: 16000Hz）

    Returns:
    --------
    numpy.ndarray
        リサンプリングされた音声信号
    """
    # リサンプリング係数の計算
    resample_ratio = target_fs / original_fs
    
    # リサンプリング後の信号長を計算
    # 元の信号長 × (目標サンプリング周波数 / 元のサンプリング周波数)
    target_length = int(len(signal_data) * resample_ratio)
    
    print(f"リサンプリング前の信号長: {len(signal_data)}サンプル")
    print(f"リサンプリング後の信号長: {target_length}サンプル")
    print(f"リサンプリング比率: {resample_ratio:.3f}")
    
    # リサンプリング
    resampled_signal = signal.resample(
        signal_data, 
        target_length
    )
    
    return resampled_signal

def check_signal_length(signal_data, fs, frame_length_ms):
    """
    入力信号の長さが窓長以上であることを確認する関数

    Parameters:
    -----------
    signal_data : numpy.ndarray
        入力音声信号
    fs : int
        サンプリング周波数
    frame_length_ms : int
        フレーム長（ミリ秒）

    Returns:
    --------
    bool
        信号長が十分な場合はTrue、そうでない場合はFalse
    """
    frame_length_samples = int(fs * frame_length_ms / 1000)
    return len(signal_data) >= frame_length_samples

def extract_spectral_envelope(signal_data, fs=16000, frame_length_ms=32, 
                            overlap=0.5, cepstrum_order=20):
    """
    ケプストラム分析によるスペクトル包絡の抽出を行う関数

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
    cepstrum_order : int, optional
        ケプストラム次数（デフォルト: 20）

    Returns:
    --------
    tuple
        (周波数軸, スペクトル包絡, スペクトル)

    Raises:
    -------
    ValueError
        入力信号の長さが窓長より短い場合
    """
    # フレーム長をサンプル数に変換
    frame_length = int(fs * frame_length_ms / 1000)
    print(f"フレーム長: {frame_length}サンプル")
    print(f"信号長: {len(signal_data)}サンプル")
    
    # 入力信号の長さチェック
    if not check_signal_length(signal_data, fs, frame_length_ms):
        raise ValueError(
            f"入力信号の長さが不足しています。"
            f"必要: {frame_length}サンプル以上, "
            f"実際: {len(signal_data)}サンプル"
        )
    
    # シフト長を計算
    shift_length = int(frame_length * (1 - overlap))
    
    # ハニング窓の作成
    window = np.hanning(frame_length)
    
    # 信号の正規化
    signal_data = signal_data / np.max(np.abs(signal_data))
    
    # フレーム分割
    num_frames = 1 + (len(signal_data) - frame_length) // shift_length
    frames = np.zeros((num_frames, frame_length))
    
    for i in range(num_frames):
        start = i * shift_length
        end = start + frame_length
        if end <= len(signal_data):
            frames[i] = signal_data[start:end] * window
    
    print(f"フレーム数: {num_frames}")
    print(f"フレームの形状: {frames.shape}")
    
    # FFTの実行
    fft_frames = np.fft.rfft(frames, axis=1)
    spectrum = np.abs(fft_frames)
    
    print(f"スペクトルの形状: {spectrum.shape}")
    
    # 周波数軸の計算
    f = np.fft.rfftfreq(frame_length, 1/fs)
    print(f"周波数軸の長さ: {len(f)}")
    
    # ケプストラム分析によるスペクトル包絡の抽出
    # 1. 対数を取る
    log_spectrum = np.log(spectrum + 1e-10)
    
    # 2. 逆フーリエ変換（ケプストラムの計算）
    cepstrum = np.fft.irfft(log_spectrum, axis=1)
    print(f"ケプストラムの形状: {cepstrum.shape}")
    
    # 3. 低次ケプストラム係数のみを使用（高次成分を0にする）
    cepstrum[:, cepstrum_order:] = 0
    
    # 4. フーリエ変換でスペクトル包絡を再構築
    spectral_envelope = np.exp(np.fft.rfft(cepstrum, axis=1))
    print(f"スペクトル包絡の形状: {spectral_envelope.shape}")
    
    # スペクトル包絡の平均を計算
    mean_envelope = np.mean(spectral_envelope, axis=0)
    mean_spectrum = np.mean(spectrum, axis=0)
    
    print(f"平均スペクトル包絡の長さ: {len(mean_envelope)}")
    print(f"平均スペクトルの長さ: {len(mean_spectrum)}")
    
    return f, mean_envelope, mean_spectrum

def find_formants(f, spectral_envelope, num_formants=2):
    """
    スペクトル包絡からフォルマントを検出する関数
    母音の一般的なフォルマント領域を考慮:
    - F1: 200-1000Hz
    - F2: 800-3000Hz

    Parameters:
    -----------
    f : numpy.ndarray
        周波数軸
    spectral_envelope : numpy.ndarray
        スペクトル包絡（1次元配列）
    num_formants : int, optional
        検出するフォルマントの数（デフォルト: 2）

    Returns:
    --------
    tuple
        (第1フォルマント周波数, 第2フォルマント周波数)
    """
    print(f"フォルマント検出時の周波数軸の長さ: {len(f)}")
    print(f"フォルマント検出時のスペクトル包絡の長さ: {len(spectral_envelope)}")
    
    # F1の検出（200-1000Hz）
    f1_mask = (f >= 200) & (f <= 1000)
    f1_region = spectral_envelope[f1_mask]
    f1_freqs = f[f1_mask]
    f1_peaks, _ = signal.find_peaks(f1_region, distance=10)
    if len(f1_peaks) > 0:
        f1_idx = f1_peaks[np.argmax(f1_region[f1_peaks])]
        f1 = f1_freqs[f1_idx]
    else:
        f1 = 500  # デフォルト値
    
    # F2の検出（800-3000Hz）
    f2_mask = (f >= 800) & (f <= 3000)
    f2_region = spectral_envelope[f2_mask]
    f2_freqs = f[f2_mask]
    f2_peaks, _ = signal.find_peaks(f2_region, distance=10)
    if len(f2_peaks) > 0:
        f2_idx = f2_peaks[np.argmax(f2_region[f2_peaks])]
        f2 = f2_freqs[f2_idx]
    else:
        f2 = 1500  # デフォルト値
    
    return f1, f2

def plot_spectral_analysis(f, spectral_envelope, spectrum, f1, f2):
    """
    スペクトル分析結果をプロットする関数

    Parameters:
    -----------
    f : numpy.ndarray
        周波数軸
    spectral_envelope : numpy.ndarray
        スペクトル包絡（1次元配列）
    spectrum : numpy.ndarray
        スペクトル（1次元配列）
    f1 : float
        第1フォルマント周波数
    f2 : float
        第2フォルマント周波数
    """
    print(f"プロット時の周波数軸の長さ: {len(f)}")
    print(f"プロット時のスペクトル包絡の長さ: {len(spectral_envelope)}")
    print(f"プロット時のスペクトルの長さ: {len(spectrum)}")
    
    # プロットの設定
    plt.figure(figsize=(12, 6))
    
    # スペクトルとスペクトル包絡のプロット
    plt.plot(f, 20 * np.log10(spectrum + 1e-10), 'b-', alpha=0.5, label='スペクトル')
    plt.plot(f, 20 * np.log10(spectral_envelope + 1e-10), 'r-', label='スペクトル包絡')
    
    # フォルマントの表示
    plt.axvline(x=f1, color='g', linestyle='--', label=f'F1: {f1:.1f} Hz')
    plt.axvline(x=f2, color='m', linestyle='--', label=f'F2: {f2:.1f} Hz')
    
    # 母音のフォルマント領域の表示
    plt.axvspan(200, 1000, color='g', alpha=0.1, label='F1領域 (200-1000Hz)')
    plt.axvspan(800, 3000, color='m', alpha=0.1, label='F2領域 (800-3000Hz)')
    
    plt.title('スペクトル包絡とフォルマント')
    plt.xlabel('周波数 [Hz]')
    plt.ylabel('振幅 [dB]')
    plt.xlim(0, 4000)  # 0-4kHzの範囲を表示
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('spectral_envelope.png')
    plt.close()

def main():
    """
    メイン関数：音声ファイルの読み込み、スペクトル包絡の抽出、
    フォルマントの検出を実行
    """
    try:
        # 音声ファイルの読み込み
        input_file = "/Users/miyatahiroshi/homework/signal-processing/05/20250515172230.wav"
        signal_data, original_fs = sf.read(input_file)
        print(f"入力音声の長さ: {len(signal_data)}サンプル")
        print(f"入力音声のサンプリング周波数: {original_fs}Hz")
        
        # ステレオ音声をモノラルに変換
        if len(signal_data.shape) > 1:
            print("ステレオ音声をモノラルに変換します...")
            signal_data = np.mean(signal_data, axis=1)
            print(f"モノラル変換後の音声の長さ: {len(signal_data)}サンプル")
        
        # 16kHzにリサンプリング
        if original_fs != 16000:
            print(f"音声を{original_fs}Hzから16kHzにリサンプリングします...")
            signal_data = resample_audio(signal_data, original_fs)
            print(f"リサンプリング後の音声の長さ: {len(signal_data)}サンプル")
            fs = 16000
        else:
            fs = original_fs
        
        # スペクトル包絡の抽出
        f, spectral_envelope, spectrum = extract_spectral_envelope(
            signal_data, 
            fs=fs
        )
        
        # フォルマントの検出
        f1, f2 = find_formants(f, spectral_envelope)
        
        # 結果のプロット
        plot_spectral_analysis(f, spectral_envelope, spectrum, f1, f2)
        
        print("処理が完了しました。")
        print(f"第1フォルマント (F1): {f1:.1f} Hz")
        print(f"第2フォルマント (F2): {f2:.1f} Hz")
        print("スペクトル包絡は spectral_envelope.png に保存されました。")
        
    except ValueError as e:
        print(f"エラー: {str(e)}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
