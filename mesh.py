import numpy as np
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import interp1d

def moving_average_x(xs, values, window_x):
    """
    x 좌표 기준으로 이동평균 수행
    """
    averaged = np.zeros_like(values)
    for i, xi in enumerate(xs):
        mask = np.abs(xs - xi) <= window_x / 2
        if np.any(mask):
            averaged[i] = np.mean(values[mask])
        else:
            averaged[i] = values[i]  # fallback
    return averaged

def curvature_based_resample(xs, ys, n_points = 100, baseline_weight=1.0, eps=1e-10, window_x = 5):
    """
    곡률 기반으로 xs를 재분포하되, 곡률이 0일 때도 기본 밀도 유지.

    Parameters
    ----------
    xs : np.ndarray
        원래 x 좌표
    ys : np.ndarray
        대응하는 y 값
    n_points : int
        재분포할 점 수
    baseline_weight : float
        곡률이 0일 때 기본으로 주는 weight 비율 (기본값=1.0, 즉 곡률 없는 영역도 분할함)
    eps : float
        0 나눔 방지용 소수

    Returns
    -------
    xs_refined : np.ndarray
        곡률 기반으로 재분포된 x좌표
    """
    
    
    # 1차 및 2차 미분
    dy = np.gradient(ys, xs)
    d2y = np.gradient(dy, xs)

    # 곡률 기반 weight + 기본 weight 혼합
    curvature_weight = moving_average_x(xs, np.abs(d2y), window_x)
    weight = curvature_weight + baseline_weight + eps  # baseline 유지
    weight /= np.sum(weight)  # 정규화

    # 누적 분포 함수
    cdf = np.cumsum(weight)
    

    
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])  # normalize to [0, 1]


    #좌우 대칭
    if len(cdf) % 2 == 0:
        cdf[len(cdf)//2:] = 1 - cdf[:len(cdf)//2][::-1]
    else:
        cdf[len(cdf)//2+1:] = 1 - cdf[:len(cdf)//2][::-1]
        cdf[len(cdf)//2] = 0.5

    # 역보간으로 새 xs 생성
    inv_cdf = interp1d(cdf, xs, kind='linear', fill_value='extrapolate')
    xs_refined = inv_cdf(np.linspace(0, 1, n_points))
    

    return xs_refined




def focused_linspace(ymin, ymax, n_element, focus='center', sharpness=5.0):
    """
    집중 분포된 linspace 생성기.
    
    Parameters
    ----------
    ymin, ymax : float
        구간 경계
    n_element : int
        총 분할 수
    focus : str ['center', 'ymin', 'ymax', 'edge']
        집중할 위치 ('edge'는 양 끝)
    sharpness : float
        집중 정도 (클수록 더 집중됨)

    Returns
    -------
    y : np.ndarray
        위치 배열 (n_element 개)
    """
    u = np.linspace(-1, 1, n_element)

    if focus == 'center':
        w = 0.5 * (1 + np.tanh(sharpness * u) / np.tanh(sharpness))
    elif focus == 'ymin':
        w = (np.exp(sharpness * u) - 1) / (np.exp(sharpness) - 1)
    elif focus == 'ymax':
        w = (np.exp(sharpness * (u + 1)) - 1) / (np.exp(sharpness * 2) - 1)
    elif focus == 'edge':
        # 가운데 느슨하게, 양쪽 촘촘하게
        w = 0.5 * (1 - np.tanh(sharpness * u) / np.tanh(sharpness))
    else:
        raise ValueError("focus must be one of: 'center', 'ymin', 'ymax', 'edge'")

    return ymin + w * (ymax - ymin)


def create_alpine(xs, center_x, amplitude, half_width):
    mask = np.ones_like(xs)
    ys = amplitude/2 + amplitude/2 * np.cos((xs - center_x) * np.pi / half_width )
    mask[xs > (center_x + half_width)] = 0
    mask[xs < (center_x - half_width)] = 0
    ys = ys * mask
    return ys

def deform_top_surface(mesh, top_deformations, ny, direction = 'top'):
    """Deform the mesh while preserving vertical ordering and fixing the bottom surface
    
    Args:
        mesh: The mesh to deform
        top_deformations: Array of target deformations for each point on the top surface
        amplitude: Maximum deformation amplitude
    """
    n_layers = ny

    
    # Apply deformation to all layers except the bottom
    for i in range(n_layers):  # Exclude the bottom layer
        # Find nodes in this layer
        layer_nodes = np.arange(i, len(mesh.p[0]), ny)
        # Calculate deformation for this layer
        # Amplitude decreases linearly from top to bottom
        # The bottom layer (i = n_layers-2) will have zero deformation
        if direction == 'top':
            layer_amplitude = i / (n_layers - 1)  # Adjusted for bottom layer exclusion
            deformation = top_deformations * layer_amplitude
        else:
            layer_amplitude = (n_layers - i) / (n_layers - 1)  # Adjusted for bottom layer exclusion
            deformation = top_deformations * layer_amplitude
        
        # Apply deformation
        mesh.p[1, layer_nodes] += deformation

def merge_array(arr1, arr2):
    return np.sort(np.unique(np.concatenate([arr1, arr2])))


