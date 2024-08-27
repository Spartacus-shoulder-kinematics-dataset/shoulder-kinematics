import numpy as np
import pytest

from .utils import TestUtils

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
        12559.349289904014,
        [
            (0, -22.415784339761863),
            (1001, 21.99062121154105),
            (2000, 24.65832131430533),
            (-1, 9.08143991261139),
        ],
    ),
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
        728.7198299305273,
        [
            (0, 16.366300000000003),
            (1001, -25.037999894224672),
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
        -1650.2696000000005,
        [
            (0, 36.84060000000001),
            (30, 29.6062),
            (60, 18.136000000000003),
            (-1, 26.8936),
        ],
    ),
    "Kijima et al.": (
        24,
        [
            "scapular plane elevation",
        ],
        [
            "scapulothoracic",
        ],
        [
            1,
            2,
            3,
        ],
        306.06436262040665,
        [
            (0, -5.259839861608079),
            (1, -0.7364922144008017),
            (2, 4.471509771399723),
            (-1, 7.421436415016586),
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
        140612.18902336064,
        [
            (0, 45.986645901396685),
            (1001, -6.497411942193646),
            (40001, -15.425452598164),
            (-1, 18.710579311164597),
        ],
    ),
    # Not in the corrected data
    # "Kozono et al.": (
    #     30,
    #     ["internal-external rotation 0 degree-abducted"],
    #     ["glenohumeral"],
    #     [1, 2, 3],
    #     0,
    #     [(0, np.nan), (1, np.nan), (2, np.nan), (-1, np.nan)],
    # ),
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
        -4261.335638555229,
        [
            (0, -111.67445727166417),
            (1, -95.19328367574825),
            (2, -83.48602756759354),
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
            (-1, 11.970999999999998),
        ],
    ),
    "Matsuki et al.": (
        1152,
        [
            "scapular plane elevation",
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
        -8652.262277420363,
        [
            (0, 0.7331976380416924),
            (1, 1.3163822061108446),
            (2, 2.009989683892475),
            (-1, 27.07331661),
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
        263654.19258251064,
        [
            (0, 5.512546219515222),
            (1, 5.512549776544782),
            (2, 5.512606226422835),
            (-1, -0.44703921745960906),
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
        978.4631094843792,
        [
            (0, 23.571499999999997),
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
        1097.2068551414447,
        [
            (0, 82.24065375507112),
            (10, 81.77476016412206),
            (22, -6.769648103841728),
            (-1, 15.647227481285565),
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
        -197.69880612483263,
        [
            (0, 44.304576869953884),
            (40, -86.8033893128552),
            (65, -7.024499957334007),
            (-1, 19.2415854),
        ],
    ),
    # Add other articles here in the same format
}
transformed_data_article = [[name] + list(values) for name, values in articles_data.items()]


spartacus = TestUtils.spartacus_folder()
module = TestUtils.load_module(spartacus + "/examples/first_example.py")
confident_values = module.main()
confident_values = confident_values[confident_values["unit"] == "rad"]


# This line parameterizes the test function below
@pytest.mark.parametrize(
    "article_name,expected_shape,humeral_motions,joints,dofs,total_value,random_checks", transformed_data_article
)
def test_article_data(article_name, expected_shape, humeral_motions, joints, dofs, total_value, random_checks):
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
    expected_articles = [
        "Begon et al.",
        "Bourne et al.",
        "Chu et al.",
        "Fung et al.",
        "Gutierrez Delgado et al.",
        "Henninger et al.",
        "Karduna et al.",
        "Kijima et al.",
        "Kim et al.",
        "Ludewig et al.",
        "Matsuki et al.",
        "Matsumura et al.",
        "Moissenet et al.",
        "Oki et al.",
        "Sahara et al.",
        "Teece et al.",
        "Yoshida et al.",
    ]
    assert articles == expected_articles

    assert confident_values.shape[0] == 809241


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


def test_glenohumeral_elevation():
    """A test because all corrections are working for these moves and joint"""
    gh_elevation_confident_values = confident_values[confident_values["joint"] == "glenohumeral"]
    motions = ["scapular plane elevation", "frontal plane elevation", "sagittal plane elevation"]
    gh_elevation_confident_values = gh_elevation_confident_values[
        gh_elevation_confident_values["humeral_motion"].isin(motions)
    ]

    articles = list(gh_elevation_confident_values["article"].unique())
    expected_articles = ["Begon et al.", "Henninger et al.", "Ludewig et al.", "Moissenet et al.", "Yoshida et al."]

    assert articles == expected_articles
    assert gh_elevation_confident_values["value"].sum() == -1838764.362187335


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
        "Yoshida et al.",
    ]

    assert articles == expected_articles
    assert st_elevation_confident_values["value"].sum() == 324261.1829014884


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
    assert sc_elevation_confident_values["value"].sum() == -2059547.3751443836


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
    assert sc_elevation_confident_values["value"].sum() == 3225402.8074460393
