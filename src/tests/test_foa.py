import numpy as np
from spatial_pipeline.foa import encode_mono_to_foa

def test_front_source():
    s = np.ones(8, dtype=np.float32)
    foa = encode_mono_to_foa(s, 0.0, 0.0, convention="basic")
    W, X, Y, Z = foa.T
    assert np.allclose(W, 1.0)
    assert np.allclose(X, 1.0, atol=1e-6)
    assert np.allclose(Y, 0.0, atol=1e-6)
    assert np.allclose(Z, 0.0, atol=1e-6)