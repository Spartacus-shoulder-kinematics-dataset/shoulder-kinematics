import numpy as np
import pytest

import spartacus as sp

spartacus_dataset = sp.load()
confident_values_all = spartacus_dataset.confident_data_values
confident_values = confident_values_all[confident_values_all["unit"] == "rad"]
confident_values_trans = confident_values_all[confident_values_all["unit"] == "mm"]

# Data for each article test
articles_data = {
    ## "Article name": (
    # expected_shape i.e. number of rows
    # humeral_motions i.e. list of humeral motions
    # joints i.e. list of joints
    # dofs i.e. list of degrees of freedom
    # total_value i.e. sum of all values
    # random_checks i.e. list of tuples (index, value) to check
    # ),
    "Bourne et al.": (
        2550,
        [
            "frontal plane elevation",
            "horizontal flexion",
        ],
        [
            "scapulothoracic",
        ],
        [
            1,
            2,
            3,
        ],
        32987.067259532865,
        [
            (0, -16.3663),
            (1001, 25.037999894224672),
            (2000, 41.36033610887097),
            (-1, 17.319848672019763),
        ],
    ),
    "Chu et al.": (
        96,
        [
            "frontal plane elevation",
            "scapular plane elevation",
            "internal-external rotation 90 degree-abducted",
        ],
        [
            "scapulothoracic",
        ],
        [
            1,
            2,
            3,
        ],
        -554.7492646716876,
        [
            (0, 20.832700000000003),
            (30, 32.982835336421985),
            (60, -30.84605712625064),
            (-1, -4.961429503563281),
        ],
    ),
    "Gutierrez Delgado et al.": (
        597,
        [
            "frontal plane elevation",
            "sagittal plane elevation",
        ],
        [
            "sternoclavicular",
        ],
        [
            1,
            2,
            3,
        ],
        -1292.21443,
        [
            (0, -31.969178000000003),
            (30, -39.584772),
            (60, -49.48644399999999),
            (-1, 33.247162),
        ],
    ),
    "Fung et al.": (
        1242,
        [
            "frontal plane elevation",
            "scapular plane elevation",
            "sagittal plane elevation",
        ],
        [
            "scapulothoracic",
            "sternoclavicular",
        ],
        [
            1,
            2,
            3,
        ],
        9188.8208,
        [
            (0, 36.8406),
            (30, 29.6062),
            (60, 18.136),
            (-1, 26.8936),
        ],
    ),
    "Kijima et al.": (
        48,
        [
            "scapular plane elevation",
        ],
        [
            "glenohumeral",
            "scapulothoracic",
        ],
        [
            1,
            2,
            3,
        ],
        1149.815953912212,
        [
            (0, np.nan),
            (1, np.nan),
            (2, np.nan),
            (-1, 35.639),
        ],
    ),
    "Begon et al.": (
        19296,
        [
            "frontal plane elevation",
            "sagittal plane elevation",
            "internal-external rotation 0 degree-abducted",
            "internal-external rotation 90 degree-abducted",
        ],
        [
            "glenohumeral",
            "scapulothoracic",
            "acromioclavicular",
            "sternoclavicular",
        ],
        [
            1,
            2,
            3,
        ],
        11186.46262507524,
        [
            (0, 22.415784339761892),
            (1001, -21.990621211541065),
            (2000, -24.65832131430533),
            (-1, 9.08143991261139),
        ],
    ),
    "Henninger et al.": (
        77472,
        [
            "frontal plane elevation",
            "scapular plane elevation",
            "sagittal plane elevation",
            "internal-external rotation 0 degree-abducted",
            "internal-external rotation 90 degree-abducted",
        ],
        [
            "glenohumeral",
            "scapulothoracic",
        ],
        [
            1,
            2,
            3,
        ],
        148532.1890233608,
        [
            (0, 45.9866459013967),
            (1001, -6.49741194219359),
            (40001, -15.425452598164),
            (-1, 18.7105793111646),
        ],
    ),
    "Kozono et al.": (
        30,
        [
            "scapular plane elevation",
        ],
        [
            "glenohumeral",
        ],
        [
            1,
            2,
            3,
        ],
        392.947,
        [
            (0, np.nan),
            (1, np.nan),
            (2, np.nan),
            (-1, 59.076),
        ],
    ),
    "Ludewig et al.": (
        684,
        [
            "frontal plane elevation",
            "scapular plane elevation",
            "sagittal plane elevation",
        ],
        [
            "glenohumeral",
            "scapulothoracic",
            "acromioclavicular",
            "sternoclavicular",
        ],
        [
            1,
            2,
            3,
        ],
        -3739.1000000000004,
        [
            (0, -8.6),
            (1, -12.0),
            (2, -15.5),
            (-1, 25.0),
        ],
    ),
    "Matsumura et al.": (
        99,
        [
            "frontal plane elevation",
            "scapular plane elevation",
            "sagittal plane elevation",
        ],
        [
            "scapulothoracic",
        ],
        [
            1,
            2,
            3,
        ],
        615.5314639526298,
        [
            (0, 23.068),
            (20, 31.07429794520548),
            (60, -8.572884301488248),
            (-1, 11.971),
        ],
    ),
    "Matsuki et al.": (
        1728,
        [
            "scapular plane elevation",
        ],
        [
            "scapulothoracic",
            "glenohumeral",
            "sternoclavicular",
        ],
        [
            1,
            2,
            3,
        ],
        -15.539620890000151,
        [
            (0, -13.69204545),
            (1, -17.77429908),
            (2, -21.54803033),
            (-1, -27.07331661),
        ],
    ),
    "Moissenet et al.": (
        705264,
        [
            "frontal plane elevation",
            "sagittal plane elevation",
            "horizontal flexion",
            "internal-external rotation 0 degree-abducted",
        ],
        [
            "glenohumeral",
            "scapulothoracic",
            "acromioclavicular",
            "sternoclavicular",
        ],
        [
            1,
            2,
            3,
        ],
        520966.64583027613,
        [
            (0, -26.1343382053074),
            (1, -26.1347999073885),
            (2, -26.1351365255401),
            (-1, -0.447039217459609),
        ],
    ),
    "Oki et al.": (
        354,
        [
            "frontal plane elevation",
            "sagittal plane elevation",
            "horizontal flexion",
        ],
        [
            "scapulothoracic",
            "sternoclavicular",
        ],
        [
            1,
            2,
            3,
        ],
        978.4631094843794,
        [
            (0, 23.5715),
            (100, -44.2176),
            (200, -25.290555552753258),
            (-1, 31.7351),
        ],
    ),
    "Teece et al.": (
        39,
        [
            "scapular plane elevation",
        ],
        [
            "acromioclavicular",
        ],
        [
            1,
            2,
            3,
        ],
        1061.3874465806718,
        [
            (0, 67.4405),
            (10, 69.0854773140717),
            (22, 6.135648345805435),
            (-1, 13.37),
        ],
    ),
    "Yoshida et al.": (
        84,
        [
            "sagittal plane elevation",
        ],
        [
            "glenohumeral",
            "scapulothoracic",
        ],
        [
            1,
            2,
            3,
        ],
        -29.895498830000026,
        [
            (0, -2.8862207),
            (40, 52.1803361),
            (65, 11.6394466),
            (-1, 19.2415854),
        ],
    ),
    # Add other articles here in the same format
}
transformed_data_article = [[name] + list(values) for name, values in articles_data.items()]


# This line parameterizes the test function below
@pytest.mark.parametrize(
    "article_name,expected_shape,humeral_motions,joints,dofs,total_value,random_checks", transformed_data_article
)
def test_article_data_no_correction(
    article_name, expected_shape, humeral_motions, joints, dofs, total_value, random_checks
):
    data = confident_values[confident_values["article"] == article_name]
    print_data(data, random_checks)

    for motion in humeral_motions:
        assert motion in data["humeral_motion"].unique()
    assert len(data["humeral_motion"].unique()) == len(humeral_motions)

    for joint in joints:
        assert joint in data["joint"].unique()
    assert len(data["joint"].unique()) == len(joints)

    for dof in dofs:
        assert dof in data["degree_of_freedom"].unique()
    assert len(data["degree_of_freedom"].unique()) == len(dofs)

    for idx, value in random_checks:
        np.testing.assert_almost_equal(data["value"].iloc[idx], value)

    assert data.shape[0] == expected_shape

    np.testing.assert_almost_equal(data["value"].sum(), total_value, decimal=10)


def test_number_of_articles():
    # Check number of unique articles after processing all
    articles = list(confident_values["article"].unique())
    experted_articles = [
        "Begon et al.",
        "Bourne et al.",
        "Chu et al.",
        "Fung et al.",
        "Gutierrez Delgado et al.",
        "Henninger et al.",
        "Karduna et al.",
        "Kijima et al.",
        "Kim et al.",
        "Kozono et al.",
        "Ludewig et al.",
        "Matsuki et al.",
        "Matsumura et al.",
        "Moissenet et al.",
        "Oki et al.",
        "Sahara et al.",
        "Sugi et al.",
        "Teece et al.",
        "Yoshida et al.",
    ]
    assert articles == experted_articles

    assert confident_values.shape[0] == 809949


def print_data(data, random_checks):
    print("")
    print("FORMATTED DATA:")
    print(f'"{data["article"].unique()[0]}": (')
    print(f"\t {data.shape[0]},")
    print(f'\t [{"".join(f"\'{motion}\', " for motion in data["humeral_motion"].unique())}],')
    print(f'\t [{"".join(f"\'{joint}\', " for joint in data["joint"].unique())}],')
    print(f"\t [{"".join(f"{dof}, " for dof in data["degree_of_freedom"].unique())}],")
    print(f"\t {data["value"].sum()},")
    print(f"\t [")
    for idx, value in random_checks:
        print(f"\t\t {idx, data.iloc[idx]['value']},")
    print(f"\t ],")
    print(f"),\n")


def test_number_of_articles_translation():
    # Check number of unique articles after processing all
    articles = list(confident_values_trans["article"].unique())
    experted_articles = ["Henninger et al.", "Kozono et al.", "Moissenet et al."]
    assert articles == experted_articles


def test_glenohumeral_elevation():
    """A test because all corrections are working for these moves and joint"""
    gh_elevation_confident_values = confident_values[confident_values["joint"] == "glenohumeral"]
    motions = ["scapular plane elevation", "frontal plane elevation", "sagittal plane elevation"]
    gh_elevation_confident_values = gh_elevation_confident_values[
        gh_elevation_confident_values["humeral_motion"].isin(motions)
    ]

    articles = list(gh_elevation_confident_values["article"].unique())
    expected_articles = [
        "Begon et al.",
        "Henninger et al.",
        "Kijima et al.",
        "Kozono et al.",
        "Ludewig et al.",
        "Matsuki et al.",
        "Moissenet et al.",
        "Yoshida et al.",
    ]

    assert articles == expected_articles
    assert gh_elevation_confident_values["value"].sum() == -1541223.3559859248


def test_scapulothoracic_elevation():
    """A test because all corrections are working for these moves and joint"""
    st_elevation_confident_values = confident_values[confident_values["joint"] == "scapulothoracic"]
    motions = ["scapular plane elevation", "frontal plane elevation", "sagittal plane elevation"]
    st_elevation_confident_values = st_elevation_confident_values[
        st_elevation_confident_values["humeral_motion"].isin(motions)
    ]

    articles = list(st_elevation_confident_values["article"].unique())
    expected_articles = [
        "Begon et al.",
        "Bourne et al.",
        "Chu et al.",
        "Fung et al.",
        "Henninger et al.",
        "Karduna et al.",
        "Kijima et al.",
        "Kim et al.",
        "Ludewig et al.",
        "Matsuki et al.",
        "Matsumura et al.",
        "Moissenet et al.",
        "Oki et al.",
        "Sugi et al.",
        "Yoshida et al.",
    ]

    assert articles == expected_articles
    assert st_elevation_confident_values["value"].sum() == 330831.9187237749


def test_sternoclavicular_elevation():
    """A test because all corrections are working for these moves and joint"""
    sc_elevation_confident_values = confident_values[confident_values["joint"] == "sternoclavicular"]
    motions = ["scapular plane elevation", "frontal plane elevation", "sagittal plane elevation"]
    sc_elevation_confident_values = sc_elevation_confident_values[
        sc_elevation_confident_values["humeral_motion"].isin(motions)
    ]

    articles = list(sc_elevation_confident_values["article"].unique())
    expected_articles = [
        "Begon et al.",
        "Fung et al.",
        "Gutierrez Delgado et al.",
        "Ludewig et al.",
        "Matsuki et al.",
        "Moissenet et al.",
        "Oki et al.",
        "Sahara et al.",
    ]

    assert articles == expected_articles
    assert sc_elevation_confident_values["value"].sum() == -1964701.8503394504


def test_acromioclavicular_elevation():
    """A test because all corrections are working for these moves and joint"""
    sc_elevation_confident_values = confident_values[confident_values["joint"] == "acromioclavicular"]
    motions = ["scapular plane elevation", "frontal plane elevation", "sagittal plane elevation"]
    sc_elevation_confident_values = sc_elevation_confident_values[
        sc_elevation_confident_values["humeral_motion"].isin(motions)
    ]

    articles = list(sc_elevation_confident_values["article"].unique())
    expected_articles = [
        "Begon et al.",
        "Ludewig et al.",
        "Moissenet et al.",
        "Sahara et al.",
        "Teece et al.",
    ]

    assert articles == expected_articles
    assert sc_elevation_confident_values["value"].sum() == 3134090.0286449315
