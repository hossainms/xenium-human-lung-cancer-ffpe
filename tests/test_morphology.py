import numpy as np
import pandas as pd

from ist_analysis import morphology as mo


def test_patches_are_centred_and_padded():
    img = np.full((100, 200, 3), 200, dtype=np.uint8)
    img[40, 120] = [1, 2, 3]                                          # one marked pixel at x = 10 + 120 * 0.5 um, y = 5 + 40 * 0.5
    origin, res = (10.0, 5.0), 0.5
    p = mo.patches(img, np.array([[70.0, 25.0], [10.0, 5.0]]), origin, res, size_px=8)
    assert p.shape == (2, 8, 8, 3)
    assert (p[0, 4, 4] == [1, 2, 3]).all()                            # the point sits at the patch centre
    assert (p[1, :4, :4] == 255).all() and (p[1, 4:, 4:] == 200).all()   # corner point: white padding outside the image


def test_balanced_sample_caps_each_label():
    labels = pd.Series(["a"] * 100 + ["b"] * 10 + ["c"] * 3)
    idx = mo.balanced_sample(labels, n_per=20, seed=0)
    counts = labels.iloc[idx].value_counts()
    assert counts.to_dict() == {"a": 20, "b": 10, "c": 3}
    assert (np.diff(idx) > 0).all()                                   # sorted, no duplicates
    assert (mo.balanced_sample(labels, 20, seed=0) == idx).all()      # reproducible


def test_neighbourhood_composition_excludes_the_cell_itself():
    xy_all = np.array([[0, 0], [5, 0], [0, 5], [100, 100]], dtype=float)
    lineage = pd.Series(["T", "Epi", "Epi", "T"])
    comp = mo.neighbourhood_composition(xy_all, lineage, xy_all[:1], radius_um=10, lineages=["Epi", "T"])
    assert np.allclose(comp, [[1.0, 0.0]])                            # only the two epithelial neighbours count


def test_cv_classify_splits_by_block_and_learns_signal(rng):
    n = 600
    y = rng.choice(["x", "y", "z"], n)
    X = rng.normal(size=(n, 10)) + 3 * pd.get_dummies(y).to_numpy().repeat(4, axis=1)[:, :10]
    groups = rng.integers(0, 12, n)
    pred = mo.cv_classify(X, y, groups, n_splits=4)
    s = mo.scores(y, pred)
    assert s["balanced accuracy"] > 0.9 and np.isclose(s["chance (balanced)"], 1 / 3)
    assert set(mo.recall_by_class(y, pred).index) == {"x", "y", "z"}


def test_cv_gene_r_separates_signal_from_noise(rng):
    n = 800
    X = rng.normal(size=(n, 20))
    Y = np.column_stack([X[:, 0] * 2 + rng.normal(scale=0.1, size=n), rng.normal(size=n)])
    r = mo.cv_gene_r(X, Y, rng.integers(0, 10, n), alphas=(1e-2, 1, 1e2))
    assert r[0] > 0.95 and abs(r[1]) < 0.15


def test_neighbourhood_mean_includes_the_cell():
    xy = np.array([[0, 0], [3, 0], [100, 0]], dtype=float)
    v = np.array([[1.0], [3.0], [10.0]])
    assert np.allclose(mo.neighbourhood_mean(xy, v, radius_um=5).ravel(), [2, 2, 10])
    import scipy.sparse as sp

    assert np.allclose(mo.neighbourhood_mean(xy, sp.csr_matrix(v), radius_um=5, rows=np.array([2, 0])).ravel(), [10, 2])


def test_stain_features_match_whole_patch_summary():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    pts = np.array([[10.0, 10.0], [20.0, 15.0], [5.0, 25.0]])
    f = mo.stain_features(img, pts, (0, 0), 0.5, size_px=16, stride=1, batch=2)
    assert f.shape == (3, 12)
    assert np.allclose(f, mo.stain_summary(mo.patches(img, pts, (0, 0), 0.5, size_px=16)), atol=1e-5)


def test_tiles_and_majority():
    xy = np.array([[1, 1], [2, 2], [3, 3], [150, 1], [151, 2]], dtype=float)
    centres, tile = mo.tiles(xy, tile_um=100, min_cells=3)
    assert np.allclose(centres, [[50, 50]]) and list(tile) == [0, 0, 0, -1, -1]
    maj = mo.majority(pd.Series(["a", "b", "a", "c", "c"]), tile, len(centres))
    assert list(maj) == ["a"]


def test_embed_centre_tokens_pick_the_patch_centre():
    """A stand-in model whose patch tokens encode their own grid position (row, col): the centre 2 x 2 average of
    a 14 x 14 grid is (6.5, 6.5), and the CLS token passes through unchanged."""
    import torch

    class Out:
        def __init__(self, h):
            self.last_hidden_state = h

    class Fake:
        def __call__(self, pixel_values):
            n, g = len(pixel_values), 14
            rows, cols = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
            tokens = torch.stack([rows, cols], dim=-1).reshape(1, g * g, 2).float().repeat(n, 1, 1)
            cls = torch.full((n, 1, 2), -1.0)
            return Out(torch.cat([cls, tokens], dim=1))

    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    cls, centre = mo.embed(Fake(), "cpu", torch.float32, img, np.array([[50.0, 50.0], [60.0, 40.0], [70.0, 70.0]]),
                           (0, 0), 0.5, batch=2, centre=2)
    assert cls.shape == (3, 2) and np.allclose(cls, -1)
    assert np.allclose(centre, 6.5)
    assert mo.embed(Fake(), "cpu", torch.float32, img, np.array([[50.0, 50.0]]), (0, 0), 0.5).shape == (1, 2)


def test_random_split_is_available_for_the_leakage_comparison(rng):
    y = np.repeat(["a", "b"], 50)
    X = np.c_[(y == "a").astype(float) * 3 + rng.normal(size=100), rng.normal(size=100)]
    assert mo.scores(y, mo.cv_classify(X, y, None))["balanced accuracy"] > 0.9
