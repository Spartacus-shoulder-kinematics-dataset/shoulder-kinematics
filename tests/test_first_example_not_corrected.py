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
            (0, -16.366300000000003),
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
        48,
        ["scapular plane elevation"],
        ["glenohumeral", "scapulothoracic"],
        [1, 2, 3],
        306.06436262040665,
        [(0, np.nan), (1, np.nan), (2, np.nan), (-1, 7.421436415016586)],
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
        246412.94812088698,
        [
            (0, 17.262926453200514),
            (1001, -119.102109559619),
            (2000, -23.049162405300553),
            (-1, 12.282290646030905),
        ],
    ),
    "Henninger et al.": (
        80862,
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
        1006110.1030133099,
        [
            (0, -134.0133540986033),
            (1001, -14.755523818348605),
            (40001, -122.66448550507789),
            (-1, 18.710579311164597),
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
        0.0,
        [
            (0, np.nan),
            (1, np.nan),
            (2, np.nan),
            (-1, np.nan),
        ],
    ),
    "Ludewig et al.": (
        684,
        ["frontal plane elevation", "scapular plane elevation", "sagittal plane elevation"],
        ["glenohumeral", "scapulothoracic", "acromioclavicular", "sternoclavicular"],
        [1, 2, 3],
        -4175.255742861286,
        [(0, -42.66781099811504), (1, -41.373669548197405), (2, -40.171439291048856), (-1, 25.0)],
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
        -558.4569560038939,
        [
            (0, -23.068),
            (20, -31.07429794520548),
            (60, 8.572884301488248),
            (-1, 11.970999999999998),
        ],
    ),
    "Matsuki et al.": (
        1836,
        ["scapular plane elevation"],
        ["scapulothoracic", "glenohumeral", "sternoclavicular"],
        [1, 2, 3],
        -18922.510132213356,
        [(0, 0.7331976380416924), (1, 1.3163822061108446), (2, 2.009989683892475), (-1, np.nan)],
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
        1665134.6062432944,
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
        2343.2226332288533,
        [
            (0, -23.571499999999997),
            (100, -44.2176),
            (200, 25.290555552753258),
            (-1, 31.7351),
        ],
    ),
    "Teece et al.": (
        39,
        ["scapular plane elevation"],
        ["acromioclavicular"],
        [1, 2, 3],
        1070.5063803329367,
        [(0, 53.208998323357854), (10, 55.34423815737802), (22, 12.65697336292887), (-1, 21.697663050779525)],
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
        551.3161398751674,
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

    assert confident_values.shape[0] == 812850


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
    experted_articles = ["Henninger et al.", "Kozono et al."]
    assert articles == experted_articles
