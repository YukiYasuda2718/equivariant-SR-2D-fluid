import numpy as np

# Refs)
# https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html#numpy.fft.fftshift
# https://qiita.com/hotta_hideyuki/items/f06aa0d0a58a28cdb055


def get_wavenumber(idx: int, total_num: int) -> int:
    if idx <= total_num // 2:
        return idx
    return idx - total_num


def calc_energy_spectrum(Q: np.ndarray) -> np.ndarray:
    """
    E(k) = 1/2 \sum |q|^2 / (k^2 + l^2) k d\theta
    assuming the domain is [0, 2 \pi] x [0, 2\pi]
    """
    S = np.fft.fft2(Q)  # spectral data
    ene_spectrum = np.zeros(sum(S.shape))

    max_idx = 0
    for i in range(S.shape[0]):
        kx = get_wavenumber(i, S.shape[0])
        for j in range(S.shape[1]):
            ky = get_wavenumber(j, S.shape[1])

            k2 = kx ** 2 + ky ** 2
            if k2 == 0:
                continue  # kx and ky == 0, so zero-division will occur

            idx = int(np.sqrt(k2) + 0.5)
            ene_spectrum[idx] += np.abs(S[i, j]) ** 2 / k2
            if idx > max_idx:
                max_idx = idx

    # The denominator is a normalization constant due to numpy fft
    return 0.5 * ene_spectrum[:max_idx] / (S.shape[0] * S.shape[1]) ** 2


def calc_energy_spectrum_from_uv(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    E(k) = 1/2 \sum (|U|^2 + |V|^2) k d\theta
    assuming the domain is [0, 2 \pi] x [0, 2\pi]
    """
    assert U.shape == V.shape
    SU = np.fft.fft2(U, axes=(-2, -1))  # spectral data
    SV = np.fft.fft2(V, axes=(-2, -1))  # spectral data
    ene_spectrum = np.zeros(SU.shape[:-2] + (sum(SU.shape[-2:]),))

    max_idx = 0
    for i in range(SU.shape[-2]):
        kx = get_wavenumber(i, SU.shape[-2])
        for j in range(SU.shape[-1]):
            ky = get_wavenumber(j, SU.shape[-1])

            k2 = kx ** 2 + ky ** 2
            idx = int(np.sqrt(k2) + 0.5)

            ene_spectrum[..., idx] += np.abs(SU[..., i, j]) ** 2 + np.abs(SV[..., i, j]) ** 2
            if idx > max_idx:
                max_idx = idx

    # The denominator is a normalization constant due to numpy fft
    return 0.5 * ene_spectrum[..., :max_idx] / (SU.shape[-2] * SU.shape[-1]) ** 2


def calc_enstrophy_spectrum(Q: np.ndarray) -> np.ndarray:
    """
    Z(k) = 1/2 \sum |q|^2 k d\theta
    assuming the domain is [0, 2 \pi] x [0, 2\pi]
    """
    S = np.fft.fft2(Q, axes=(-2, -1))  # spectral data
    ens_spectrum = np.zeros(S.shape[:-2] + (sum(S.shape[-2:]),))

    max_idx = 0
    for i in range(S.shape[-2]):
        kx = get_wavenumber(i, S.shape[-2])
        for j in range(S.shape[-1]):
            ky = get_wavenumber(j, S.shape[-1])

            k2 = kx ** 2 + ky ** 2
            if k2 == 0:
                continue  # kx and ky == 0

            idx = int(np.sqrt(k2) + 0.5)
            ens_spectrum[..., idx] += np.abs(S[..., i, j]) ** 2
            if idx > max_idx:
                max_idx = idx

    # The denominator is a normalization constant due to numpy fft
    return 0.5 * ens_spectrum[..., :max_idx] / (S.shape[-2] * S.shape[-1]) ** 2


def calc_Re(vortex_field: np.ndarray, nu: float) -> float:
    ene_spec = calc_energy_spectrum(vortex_field[:, :])
    ens_spec = calc_enstrophy_spectrum(vortex_field[:, :])
    u = np.sqrt(np.sum(ene_spec))  # sqrt of spatial average of kinetic energy
    l = u / np.sqrt(np.sum(ens_spec))  # integral length scale
    re = u * l / nu
    return re


def calc_stream_function(Z: np.ndarray) -> np.ndarray:
    S = np.fft.fft2(Z)  # spectral data
    PSI = np.zeros_like(S)
    for i in range(S.shape[0]):
        kx = get_wavenumber(i, S.shape[0])
        for j in range(S.shape[1]):
            ky = get_wavenumber(j, S.shape[1])

            k2 = np.abs(kx ** 2 + ky ** 2)
            if k2 == 0:
                PSI[i, j] = 0.0
            else:
                PSI[i, j] = -S[i, j] / k2  # stream function

    return np.real(np.fft.ifft2(PSI))


def calc_velocity(Z: np.ndarray, is_xcomponent: bool) -> np.ndarray:
    S = np.fft.fft2(Z, axes=(-2, -1))  # spectral data
    V = np.zeros_like(S)
    for i in range(S.shape[-2]):
        kx = get_wavenumber(i, S.shape[-2])
        for j in range(S.shape[-1]):
            ky = get_wavenumber(j, S.shape[-1])

            k2 = np.abs(kx ** 2 + ky ** 2)
            if k2 == 0:
                V[..., i, j] = 0.0
                continue  # kx and ky == 0

            psi = -S[..., i, j] / k2  # stream function

            if is_xcomponent:
                V[..., i, j] = -(ky * 1j) * psi
            else:
                V[..., i, j] = (kx * 1j) * psi
    return np.real(np.fft.ifft2(V, axes=(-2, -1)))


def calc_derivative(G: np.ndarray, is_x: bool) -> np.ndarray:
    S = np.fft.fft2(G, axes=(-2, -1))  # spectral data
    D = np.zeros_like(S)
    for i in range(S.shape[-2]):
        kx = get_wavenumber(i, S.shape[-2])
        for j in range(S.shape[-1]):
            ky = get_wavenumber(j, S.shape[-1])

            if is_x:
                D[..., i, j] = (kx * 1j) * S[..., i, j]
            else:
                D[..., i, j] = (ky * 1j) * S[..., i, j]
    return np.real(np.fft.ifft2(D, axes=(-2, -1)))


def reflect(Z: np.ndarray) -> np.ndarray:
    assert len(Z.shape) == 2

    reflected = np.zeros((Z.shape[0], (Z.shape[1] - 1) * 2))
    reflected[:, : Z.shape[1]] = Z
    reflected[:, Z.shape[1] :] = -Z[:, ::-1][:, 1:-1]

    return reflected


def calc_velocity_using_sine_transform(Z: np.ndarray, is_xcomponent: bool) -> np.ndarray:
    reflected = reflect(Z)
    velocity = calc_velocity(reflected, is_xcomponent)
    velocity = velocity[:, : Z.shape[1]]
    assert velocity.shape == Z.shape
    return velocity


def reflect_velocity(data: np.ndarray, is_u: bool) -> np.ndarray:

    reflected = np.zeros(data.shape[:-2] + (data.shape[-2], (2 * data.shape[-1] - 1)))
    reflected[..., : data.shape[-1]] = data

    if is_u:
        reflected[..., data.shape[-1] :] = data[..., ::-1][..., 1:]  # for cosine transform
    else:
        reflected[..., data.shape[-1] :] = -data[..., ::-1][..., 1:]  # for sine transform

    return reflected


def calc_vorticity_using_sine_cosine_transform(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert u.shape == v.shape
    assert len(u.shape) == 2

    _u = reflect_velocity(u, is_u=True)
    _v = reflect_velocity(v, is_u=False)
    _z = calc_derivative(_v, is_x=True) - calc_derivative(_u, is_x=False)
    z = _z[:, : u.shape[1]]

    assert z.shape == u.shape

    return z


def calc_vorticity(uv):
    assert uv.shape[-3] == 2  # u and v comnmponents
    u, v = uv[..., 0, :, :], uv[..., 1, :, :]
    z = calc_derivative(v, is_x=True) - calc_derivative(u, is_x=False)
    return z


def calc_energy_spectrum_from_uv_after_scaling(uv: np.ndarray, config: dict) -> np.ndarray:
    assert len(uv.shape) == 3
    assert uv.shape[0] == 2

    _uv = uv * config["data"]["velocity_std"] + config["data"]["velocity_mean"]
    ene_spec_from_V = calc_energy_spectrum_from_uv(_uv[0], _uv[1])

    return ene_spec_from_V


def calc_energy_spectrum_from_z_after_scaling(z: np.ndarray, config: dict) -> np.ndarray:
    assert len(z.shape) == 2

    _z = z * config["data"]["vortex_std"] + config["data"]["vortex_mean"]
    ene_spec_from_Z = calc_energy_spectrum(_z)

    return ene_spec_from_Z


def calc_enstrophy_spectrum_from_uv_after_scaling(uv: np.ndarray, config: dict) -> np.ndarray:
    assert len(uv.shape) == 3
    assert uv.shape[0] == 2

    _uv = uv * config["data"]["velocity_std"] + config["data"]["velocity_mean"]
    z = calc_vorticity(_uv)
    ens_spec = calc_enstrophy_spectrum(z)

    return ens_spec


def calc_enstrophy_spectrum_from_z_after_scaling(z: np.ndarray, config: dict) -> np.ndarray:
    assert len(z.shape) == 2

    _z = z * config["data"]["vortex_std"] + config["data"]["vortex_mean"]
    ens_spec = calc_enstrophy_spectrum(_z)

    return ens_spec


def calc_spectrum_set_from_z_after_scaling(z: np.ndarray, config: dict) -> dict:
    assert len(z.shape) == 2

    _z = z * config["data"]["vortex_std"] + config["data"]["vortex_mean"]

    ens_spec = calc_enstrophy_spectrum(_z)
    ene_spec = calc_energy_spectrum(_z)
    assert len(ens_spec) == len(ene_spec)

    ks = np.arange(len(ene_spec))

    return {
        "energy_spectrum": ene_spec,
        "enstrophy_spectrum": ens_spec,
        "energy_spectrum_xk": ene_spec * ks,
        "enstrophy_spectrum_xk": ens_spec * ks,
    }


def calc_spectrum_set_from_uv_after_scaling(uv: np.ndarray, config: dict) -> dict:
    assert uv.shape[-3] == 2  # u and v components

    _uv = uv * config["data"]["velocity_std"] + config["data"]["velocity_mean"]
    z = calc_vorticity(_uv)
    ens_spec = calc_enstrophy_spectrum(z)
    ene_spec = calc_energy_spectrum_from_uv(_uv[..., 0, :, :], _uv[..., 1, :, :])

    assert ens_spec.shape == ene_spec.shape

    ks = np.arange(ene_spec.shape[-1])

    return {
        "energy_spectrum": ene_spec,
        "enstrophy_spectrum": ens_spec,
        "energy_spectrum_xk": ene_spec * ks,
        "enstrophy_spectrum_xk": ens_spec * ks,
    }


def calc_spectrum_set_from_z_after_scaling_using_sine_transform(
    z: np.ndarray, config: dict
) -> dict:
    assert len(z.shape) == 2

    _z = z * config["data"]["vortex_std"] + config["data"]["vortex_mean"]
    _z = reflect(_z)

    ens_spec = calc_enstrophy_spectrum(_z) / 4.0
    ene_spec = calc_energy_spectrum(_z) / 4.0
    assert len(ens_spec) == len(ene_spec)

    ks = np.arange(len(ene_spec))

    return {
        "energy_spectrum": ene_spec,
        "enstrophy_spectrum": ens_spec,
        "energy_spectrum_xk": ene_spec * ks,
        "enstrophy_spectrum_xk": ens_spec * ks,
    }


def calc_spectrum_set_from_uv_after_scaling_using_sine_cosine_transform(
    uv: np.ndarray, config: dict
) -> dict:
    assert uv.shape[-3] == 2  # u and v components

    _uv = np.zeros_like(uv)

    _uv[..., 0, :, :] = uv[..., 0, :, :] * config["data"]["u_std"] + config["data"]["u_mean"]
    _uv[..., 1, :, :] = uv[..., 1, :, :] * config["data"]["v_std"] + config["data"]["v_mean"]
    _u = reflect_velocity(_uv[..., 0, :, :], is_u=True)
    _v = reflect_velocity(_uv[..., 1, :, :], is_u=False)
    _uv = np.stack([_u, _v], axis=-3)

    assert _uv.shape[-3] == 2  # u and v components

    z = calc_vorticity(_uv)
    ens_spec = calc_enstrophy_spectrum(z) / 4.0
    ene_spec = calc_energy_spectrum_from_uv(_uv[..., 0, :, :], _uv[..., 1, :, :]) / 4.0

    assert ens_spec.shape == ene_spec.shape

    ks = np.arange(ene_spec.shape[-1])

    return {
        "energy_spectrum": ene_spec,
        "enstrophy_spectrum": ens_spec,
        "energy_spectrum_xk": ene_spec * ks,
        "enstrophy_spectrum_xk": ens_spec * ks,
    }
