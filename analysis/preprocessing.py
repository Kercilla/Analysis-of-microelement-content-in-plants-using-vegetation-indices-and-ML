import numpy as np
from scipy.signal import savgol_filter

_trapz = getattr(np, "trapezoid", None) or np.trapz


def sg_smooth(spectra, window=11, poly=2):
    if window % 2 == 0:
        window += 1
    return savgol_filter(spectra, window, poly, axis=1)


def sg_derivative(spectra, order=1, window=11, poly=2):
    if window % 2 == 0:
        window += 1
    poly = max(poly, order)
    return savgol_filter(spectra, window, poly, deriv=order, axis=1)


def snv(spectra):
    mu = spectra.mean(axis=1, keepdims=True)
    sd = spectra.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (spectra - mu) / sd


def continuum_removal(spectra, wavelengths):
    from scipy.spatial import ConvexHull
    out = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        y = spectra[i]
        pts = np.column_stack([wavelengths, y])
        try:
            hull = ConvexHull(pts)
            verts = sorted(set(hull.vertices))
            upper = np.interp(wavelengths, wavelengths[verts], y[verts])
            upper[upper == 0] = 1.0
            out[i] = y / upper
        except Exception:
            out[i] = y
    return out


def remove_noisy_bands(spectra, wavelengths, ranges=None):
    if ranges is None:
        ranges = [(925, 975)]
    mask = np.ones(len(wavelengths), dtype=bool)
    for lo, hi in ranges:
        mask &= ~((wavelengths >= lo) & (wavelengths <= hi))
    return spectra[:, mask], wavelengths[mask]


def cwt_features(spectra, wavelengths, scales=None):
    import pywt
    if scales is None:
        scales = np.arange(2, 32, 2)
    feats = []
    for i in range(spectra.shape[0]):
        coefs, _ = pywt.cwt(spectra[i], scales, "mexh")
        feats.append(np.concatenate([
            np.abs(coefs).mean(axis=1),
            coefs.max(axis=1),
            coefs.min(axis=1),
        ]))
    names = []
    for stat in ("abs_mean", "max", "min"):
        for s in scales:
            names.append(f"cwt_{stat}_s{s}")
    return np.array(feats), names


def spectral_shape_features(spectra, wavelengths):
    n = spectra.shape[0]
    out = []
    for i in range(n):
        s = spectra[i]
        f = {}

        re_mask = (wavelengths >= 680) & (wavelengths <= 760)
        if re_mask.sum() > 3:
            wl_re = wavelengths[re_mask]
            s_re = s[re_mask]
            d = np.gradient(s_re, wl_re)
            f["rep"] = wl_re[np.argmax(d)]
            f["re_slope"] = d.max()
            f["re_area"] = _trapz(s_re, wl_re)
        else:
            f["rep"] = f["re_slope"] = f["re_area"] = np.nan

        gp_mask = (wavelengths >= 500) & (wavelengths <= 580)
        if gp_mask.sum() > 1:
            f["green_peak"] = s[gp_mask].max()
            f["green_wl"] = wavelengths[gp_mask][np.argmax(s[gp_mask])]
        else:
            f["green_peak"] = f["green_wl"] = np.nan

        i680 = np.argmin(np.abs(wavelengths - 680))
        i750 = np.argmin(np.abs(wavelengths - 750))
        i550 = np.argmin(np.abs(wavelengths - 550))
        baseline = np.interp(680, [wavelengths[i550], wavelengths[i750]],
                             [s[i550], s[i750]])
        f["chl_depth"] = baseline - s[i680]
        f["chl_norm"] = f["chl_depth"] / (baseline + 1e-8)

        nir_mask = (wavelengths >= 780) & (wavelengths <= 900)
        if nir_mask.sum() > 1:
            f["nir_mean"] = s[nir_mask].mean()
            f["nir_std"] = s[nir_mask].std()
            f["nir_slope"] = np.polyfit(wavelengths[nir_mask], s[nir_mask], 1)[0]
        else:
            f["nir_mean"] = f["nir_std"] = f["nir_slope"] = np.nan

        w_mask = (wavelengths >= 950) & (wavelengths <= 990)
        f["water_depth"] = s[nir_mask].mean() - s[w_mask].min() if nir_mask.sum() > 0 and w_mask.sum() > 0 else np.nan

        vis = (wavelengths >= 400) & (wavelengths <= 700)
        nir2 = (wavelengths >= 700) & (wavelengths <= 1000)
        if vis.sum() > 1 and nir2.sum() > 1:
            f["nir_vis_ratio"] = _trapz(s[nir2], wavelengths[nir2]) / \
                                 (_trapz(s[vis], wavelengths[vis]) + 1e-8)
        else:
            f["nir_vis_ratio"] = np.nan

        out.append(f)

    names = list(out[0].keys())
    arr = np.array([[r.get(k, np.nan) for k in names] for r in out])
    return arr, names


def mixup_augment(X, y, n_aug=None, alpha=0.3, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n_aug is None:
        n_aug = n
    ia = rng.integers(0, n, size=n_aug)
    ib = rng.integers(0, n, size=n_aug)
    lam = rng.beta(alpha, alpha, size=(n_aug, 1))
    Xm = lam * X[ia] + (1 - lam) * X[ib]
    ym = lam.ravel() * y[ia] + (1 - lam.ravel()) * y[ib]
    return np.vstack([X, Xm]), np.concatenate([y, ym])


def noise_augment(X, y, n_aug=None, noise_pct=0.02, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n_aug is None:
        n_aug = n
    idx = rng.integers(0, n, size=n_aug)
    sigma = X.std(axis=0, keepdims=True) * noise_pct
    Xn = X[idx] + rng.normal(0, 1, (n_aug, X.shape[1])) * sigma
    return np.vstack([X, Xn]), np.concatenate([y, y[idx]])


def preprocess_pipeline(spectra, wavelengths=None, steps=None, sg_window=11, sg_poly=2):
    if steps is None:
        steps = ["smooth", "deriv1", "snv"]
    X = spectra.copy().astype(float)
    wl = wavelengths.copy() if wavelengths is not None else None

    dispatch = {
        "smooth":  lambda: sg_smooth(X, sg_window, sg_poly),
        "deriv1":  lambda: sg_derivative(X, 1, sg_window, sg_poly),
        "deriv2":  lambda: sg_derivative(X, 2, sg_window, max(sg_poly, 3)),
        "snv":     lambda: snv(X),
    }

    for step in steps:
        if step in dispatch:
            X = dispatch[step]()
        elif step == "remove_noise" and wl is not None:
            X, wl = remove_noisy_bands(X, wl)
        elif step == "continuum" and wl is not None:
            X = continuum_removal(X, wl)
        else:
            raise ValueError(f"Unknown step: {step}")

    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), wl
