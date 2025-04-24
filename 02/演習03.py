"""
FIRフィルタを使用したバンドパスフィルタの設計と実装

仕様:
- 遮断周波数: 2kHz（下限）と6kHz（上限）
- サンプリング周波数: 16kHz
- フィルタ次数: M=100
- フィルタタイプ: FIRバンドパスフィルタ
- 窓関数: ハミング窓

制限事項:
- 入力音源はwav形式である必要があります
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

# 定数定義
class Constants:
    """フィルタ設計の定数クラス"""
    # フィルタパラメータ
    LOW_CUTOFF = 2000  # 下限遮断周波数 [Hz]
    HIGH_CUTOFF = 6000  # 上限遮断周波数 [Hz]
    SAMPLING_FREQ = 16000  # サンプリング周波数 [Hz]
    FILTER_ORDER = 100  # フィルタ次数
    
    # ファイルパス
    INPUT_FILE = '/Users/miyatahiroshi/homework/信号処理システム/02/white.wav'
    OUTPUT_FILE = 'output03.wav'
    
    # エラーメッセージ
    ERROR_FILE_NOT_FOUND = "エラー: 入力ファイルが見つかりません。"
    ERROR_SAMPLING_RATE = "サンプリング周波数が一致しません: 期待値={}Hz, 実際={}Hz"
    ERROR_UNEXPECTED = "予期せぬエラーが発生しました: {}"


def design_bandpass_filter(low_cutoff, high_cutoff, sampling_freq, filter_order):
    """
    バンドパスフィルタを設計する関数

    Parameters:
    -----------
    low_cutoff : float
        下限遮断周波数 [Hz]
    high_cutoff : float
        上限遮断周波数 [Hz]
    sampling_freq : float
        サンプリング周波数 [Hz]
    filter_order : int
        フィルタ次数

    Returns:
    --------
    numpy.ndarray
        フィルタ係数
    """
    # 正規化された遮断周波数を計算
    normalized_low = low_cutoff / (sampling_freq / 2)
    normalized_high = high_cutoff / (sampling_freq / 2)
    
    # FIRフィルタを設計（ハミング窓を使用）
    filter_coefficients = signal.firwin(
        filter_order + 1,  # フィルタ次数
        [normalized_low, normalized_high],  # 正規化された遮断周波数
        window='hamming',  # 窓関数
        pass_zero=False    # バンドパスフィルタの指定
    )
    
    return filter_coefficients


def apply_filter(input_signal, filter_coefficients):
    """
    フィルタを信号に適用する関数

    Parameters:
    -----------
    input_signal : numpy.ndarray
        入力信号
    filter_coefficients : numpy.ndarray
        フィルタ係数

    Returns:
    --------
    numpy.ndarray
        フィルタ適用後の信号
    """
    return signal.lfilter(filter_coefficients, 1.0, input_signal)


def validate_sampling_rate(actual_rate, expected_rate):
    """
    サンプリング周波数を検証する関数

    Parameters:
    -----------
    actual_rate : int
        実際のサンプリング周波数
    expected_rate : int
        期待するサンプリング周波数

    Raises:
    -------
    ValueError
        サンプリング周波数が一致しない場合
    """
    if actual_rate != expected_rate:
        raise ValueError(
            Constants.ERROR_SAMPLING_RATE.format(expected_rate, actual_rate)
        )


def main():
    """メイン処理"""
    try:
        # フィルタを設計
        filter_coefficients = design_bandpass_filter(
            Constants.LOW_CUTOFF,
            Constants.HIGH_CUTOFF,
            Constants.SAMPLING_FREQ,
            Constants.FILTER_ORDER
        )
        
        # 入力音源を読み込み
        input_signal, sampling_rate = sf.read(Constants.INPUT_FILE)
        
        # サンプリング周波数の確認
        validate_sampling_rate(sampling_rate, Constants.SAMPLING_FREQ)
        
        # フィルタを適用
        output_signal = apply_filter(input_signal, filter_coefficients)
        
        # 出力信号を保存
        sf.write(Constants.OUTPUT_FILE, output_signal, Constants.SAMPLING_FREQ)
        
        print(f"フィルタ処理が完了しました。{Constants.OUTPUT_FILE}を確認してください。")
        
    except FileNotFoundError:
        print(Constants.ERROR_FILE_NOT_FOUND)
    except ValueError as e:
        print(f"エラー: {str(e)}")
    except Exception as e:
        print(Constants.ERROR_UNEXPECTED.format(str(e)))


if __name__ == "__main__":
    main()
