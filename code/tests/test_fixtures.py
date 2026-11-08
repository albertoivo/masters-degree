"""Sanidade do dataset sintético usado pelos demais testes."""
import numpy as np
from PIL import Image


def test_synthetic_dataset_layout(synthetic_dataset):
    ds = synthetic_dataset
    df = ds.labels()
    assert len(df) == 9
    assert set(df['filename']) == {f for fs in ds.labeled.values() for f in fs}
    # noite nublada: has_epb vazio, cloudy=1
    cloudy = df[df['cloudy'] == 1]
    assert len(cloudy) == 3 and cloudy['has_epb'].isna().all()
    assert set(df.loc[df['cloudy'] == 0, 'has_epb']) == {0, 1}
    for name in df['filename']:
        img = Image.open(ds.images_dir / name)
        assert img.mode == 'L' and img.size == (512, 512)
    for night, names in ds.unlabeled.items():
        assert names and all((ds.processed_dir(night) / n).exists() for n in names)
        assert not set(names) & set(df['filename'])
    assert not ds.test_nights_path.exists()


def test_synthetic_images_are_deterministic(synthetic_dataset):
    name = synthetic_dataset.labeled['CA_20251022'][0]
    a = np.array(Image.open(synthetic_dataset.images_dir / name))
    b = np.array(Image.open(synthetic_dataset.processed_dir('CA_20251022') / name))
    assert np.array_equal(a, b)
