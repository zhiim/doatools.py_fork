import numpy as np
from ..core import get_noise_subspace, get_signal_subspace, find_peaks_simple
from .wideband_core import divide_wideband_into_sub, get_estimates_from_sp

C = 3e8  # wave speed

class TOPS():
    def __init__(self, array, search_grid, peak_finder=find_peaks_simple):
        self._array = array
        self._search_grid = search_grid
        self._peak_finder = peak_finder

    def _spatial_spectrum(self, signal, fs, f_start, f_end, n_fft, k):
        # divide wideband signal into frequency points
        signal_subs, freq_bins = divide_wideband_into_sub(signal, n_fft, fs,
                                                          f_start, f_end)
        num_group = signal_subs.shape[2]  # 每个频点的FFT组数
        array_location = self._array.actual_element_locations

        f_reference = freq_bins[int(freq_bins.size / 2)]  # 参考频点
        # 得到参考频点对应的频域接收信号
        matrix_x_ref = signal_subs[:, int(freq_bins.size / 2), :]
        # 对参考频点处的信号进行正交分解，得到对应的信号子空间和噪声子空间
        signal_space_ref = get_signal_subspace(
            matrix_r=matrix_x_ref @ matrix_x_ref.conj().T / num_group, k=k)

        # grid实例对应的网格点，用于遍历构造空间谱
        grids_location = self._search_grid.axes[0]

        sp = np.zeros(grids_location.size)  # 空间谱
        for i, grid_point in enumerate(grids_location):
            matrix_d = np.empty((k, 0), dtype=np.complex_)

            for j, freq in enumerate(freq_bins):
                matrix_x_f = signal_subs[:, j, :]  # 当前频点对应的频域接收信号
                # 计算当前频点对应的噪声子空间
                noise_space_f = get_noise_subspace(
                    matrix_r=matrix_x_f @ matrix_x_f.conj().T / num_group, k=k)

                # 构造变换矩阵
                matrix_phi = np.exp(1j * 2 * np.pi * (freq - f_reference) / C\
                                     * array_location * np.sin(grid_point))
                matrix_phi = np.diag(np.squeeze(matrix_phi))
                # 使用变换矩阵将参考频点的信号子空间变换到当前频点
                matrix_u = matrix_phi @ signal_space_ref

                # 构造投影矩阵，减小矩阵U中的误差
                matrix_a_f = np.exp(1j * 2 * np.pi * freq / C\
                                     * array_location * np.sin(grid_point))
                matrix_p = np.eye(self._array.size) -\
                            1 / (matrix_a_f.conj().T @ matrix_a_f) *\
                            matrix_a_f @ matrix_a_f.conj().T
                # 使用投影矩阵对矩阵U进行投影
                matrix_u = matrix_p @ matrix_u

                matrix_d = np.concatenate((matrix_d,
                                           matrix_u.conj().T @ noise_space_f),
                                           axis=1)
            # 使用矩阵D中的最小特征值构造空间谱
            _, s, _ = np.linalg.svd(matrix_d)
            sp[i] = 1 / min(s)

        return sp

    def estimate(self, signal, fs, f_start, f_end, n_fft, k,
                 return_spectrum=True):
        sp = self._spatial_spectrum(signal=signal, fs=fs, f_start=f_start,
                                    f_end=f_end, n_fft=n_fft, k=k)
        return get_estimates_from_sp(sp=sp, k=k, search_grid=self._search_grid,
                                     peak_finder=self._peak_finder,
                                     return_spectrum=return_spectrum)
