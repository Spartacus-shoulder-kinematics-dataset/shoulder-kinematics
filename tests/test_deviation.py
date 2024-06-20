import numpy as np

from spartacus import DataFolder, load_subdataset

deviation_cols = [
    "parent_d1",  # float
    "parent_d2",  # float
    "parent_d3",  # float
    "parent_d4",  # float
    "child_d1",  # float
    "child_d2",  # float
    "child_d3",  # float
    "child_d4",  # float
    "d5",  # float
    "d6",  # float
    "d7",  # float
    "total_deviation",  # float
]


def test_deviation_begon():
    data = load_subdataset(name=DataFolder.BEGON_2014)
    df = data.corrected_confident_data_values

    subdf = df[df["joint"] == "sternoclavicular"]
    np.testing.assert_almost_equal(
        subdf[deviation_cols].values[0, :], [0.9, 1.0, 0.5, 1, 0.9, 1.0, 0.5, 1, 1.0, 1.0, 1.0, 0.2025]
    )

    subdf = df[df["joint"] == "acromioclavicular"]
    np.testing.assert_almost_equal(
        subdf[deviation_cols].values[0, :], [0.9, 1.0, 0.5, 1.0, 0.9, 1.0, 0.5, 0.9, 1.0, 1.0, 1.0, 0.18225]
    )

    subdf = df[df["joint"] == "glenohumeral"]
    np.testing.assert_almost_equal(
        subdf[deviation_cols].values[0, :], [0.9, 1.0, 0.5, 0.9, 0.9, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.18225]
    )

    subdf = df[df["joint"] == "scapulothoracic"]
    np.testing.assert_almost_equal(
        subdf[deviation_cols].values[0, :], [0.9, 1.0, 0.5, 1, 0.9, 1.0, 0.5, 0.9, 1.0, 1.0, 1.0, 0.18225]
    )


def test_deviation_bourne():
    data = load_subdataset(name=DataFolder.BOURNE_2003)
    df = data.corrected_confident_data_values

    subdf = df[df["joint"] == "scapulothoracic"]
    np.testing.assert_almost_equal(
        subdf[deviation_cols].values[0, :], [0.9, 1.0, 0.5, 1.0, 0.9, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.2025]
    )


def test_deviation_chu():
    data = load_subdataset(name=DataFolder.CHU_2012)
    df = data.corrected_confident_data_values

    subdf = df[df["joint"] == "scapulothoracic"]
    np.testing.assert_almost_equal(
        subdf[deviation_cols].values[0, :], [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
    )


def test_deviation_fung():
    # not properly labeled yet (clavicle)
    print("Test not implemented yet")
    # data = load_subdataset(name=DataFolder.FUNG_2001)
    #
    # df = data.corrected_confident_data_values
    #
    # subdf = df[df["joint"] == "scapulothoracic"]
    # np.testing.assert_almost_equal(
    #     subdf[deviation_cols].values[0, :], [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
    # )

    # subdf = df[df["joint"] == "sternoclavicular"]
    # np.testing.assert_almost_equal(
    #     subdf[deviation_cols].values[0, :], [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
    # )


def test_deviation_henninger():
    data = load_subdataset(name=DataFolder.KOLZ_2020)
    df = data.corrected_confident_data_values

    subdf = df[df["joint"] == "scapulothoracic"]
    np.testing.assert_almost_equal(
        subdf[deviation_cols].values[0, :], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )

    subdf = df[df["joint"] == "glenohumeral"]
    np.testing.assert_almost_equal(
        subdf[deviation_cols].values[0, :], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )


def test_deviation_karduna():
    data = load_subdataset(name=DataFolder.MCCLURE_2001)
    df = data.corrected_confident_data_values

    subdf = df[df["joint"] == "scapulothoracic"]
    np.testing.assert_almost_equal(
        subdf[deviation_cols].values[0, :], [0.9, 1.0, 0.5, 0.9, 0.9, 1.0, 0.5, 0.9, 1.0, 1.0, 1.0, 0.164025]
    )
