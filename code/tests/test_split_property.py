"""Property 1: Dev and test splits partition the data by night.

**Validates: Requirements 1.3, 1.4, 1.5**
"""
import tempfile
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from epb_detection_v5 import load_raw_dataset, extract_night_groups

KNOWN_NIGHTS = ['CA_20251022', 'BJL_20250722', 'CA_20260624']
ABSENT_NIGHTS = ['CA_20990101', 'BJL_20000101', 'STR_20250801']


@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(test_nights=st.sets(st.sampled_from(KNOWN_NIGHTS + ABSENT_NIGHTS)),
       include_cloudy=st.booleans())
def test_dev_and_test_partition_by_night(synthetic_dataset, test_nights, include_cloudy):
    # O fixture não é alterado: a lista de noites de teste vai para um arquivo temporário próprio.
    ds = synthetic_dataset
    with tempfile.TemporaryDirectory() as tmp:
        nights_file = Path(tmp) / 'test_nights.txt'
        nights_file.write_text('# lista gerada\n' + '\n'.join(sorted(test_nights)) + '\n')

        def load(split):
            return load_raw_dataset(str(ds.images_dir), str(ds.labels_csv),
                                    include_cloudy=include_cloudy, split=split,
                                    test_nights_path=str(nights_file))

        X_dev, y_dev, df_dev = load('dev')
        X_test, y_test, df_test = load('test')
        _, y_all, df_all = load('all')

    dev, test, all_ = (set(d['filename']) for d in (df_dev, df_test, df_all))
    assert dev.isdisjoint(test)
    assert dev | test == all_
    assert len(df_dev) + len(df_test) == len(df_all)
    assert len(X_dev) == len(y_dev) == len(df_dev)
    assert len(X_test) == len(y_test) == len(df_test)

    dev_nights = set(extract_night_groups(df_dev)) if len(df_dev) else set()
    test_nights_seen = set(extract_night_groups(df_test)) if len(df_test) else set()
    assert dev_nights.isdisjoint(test_nights_seen)
    assert test_nights_seen <= test_nights
    assert dev_nights.isdisjoint(test_nights)

    # Rótulos preservados: cada quadro tem o mesmo y em dev/test e em all.
    y_by_name = dict(zip(df_all['filename'], y_all))
    for df, y in ((df_dev, y_dev), (df_test, y_test)):
        assert all(y_by_name[f] == v for f, v in zip(df['filename'], np.asarray(y)))
