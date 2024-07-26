import numpy as np
import pytest

import spartacus as sp

spartacus_dataset = sp.load()
confident_values = spartacus_dataset.confident_data_values
confident_values = confident_values[confident_values["unit"] == "rad"]

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
        ["frontal plane elevation", "horizontal flexion"],
        ["scapulothoracic"],
        [1, 2, 3],
        32987.06725953286,
        [(0, -16.3663), (1001, 21.925651818083274), (2000, -38.92291527743668), (-1, 17.319848672019763)],
    ),
    "Chu et al.": (
        96,
        ["frontal plane elevation", "scapular plane elevation", "internal-external rotation 90 degree-abducted"],
        ["scapulothoracic"],
        [1, 2, 3],
        -554.7492646716876,
        [(0, 20.8327), (30, -2.4029921148049445), (60, -8.436784721537704), (-1, -4.961429503563281)],
    ),
    "Fung et al.": (
        1242,
        ["frontal plane elevation", "scapular plane elevation", "sagittal plane elevation"],
        ["scapulothoracic"],
        [1, 2, 3],
        -1650.2695999999999,
        [(0, 36.84060000000001), (30, 29.6062), (60, 18.136000000000003), (-1, 26.8936)],
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
        [1, 2, 3],
        246412.948120887,
        [(0, 17.262926453200514), (1001, 47.33668895380996), (2000, 42.92927676074317), (-1, 12.282290646030905)],
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
        ["glenohumeral", "scapulothoracic"],
        [1, 2, 3],
        1006110.1030133098,
        [(0, -134.0133540986033), (1001, 76.9873557465161), (40001, 171.3809554730667), (-1, 18.710579311164597)],
    ),
    "Kozono et al.": (
        30,
        ["internal-external rotation 0 degree-abducted"],
        ["glenohumeral"],
        [1, 2, 3],
        392.947,
        [(0, np.nan), (1, np.nan), (2, np.nan), (-1, 59.076)],
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
        ["frontal plane elevation", "scapular plane elevation", "sagittal plane elevation"],
        ["scapulothoracic"],
        [1, 2, 3],
        -558.4569560038939,
        [(0, -23.068), (20, 32.595395315826835), (60, -0.8599417044686808), (-1, 11.971)],
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
        ["glenohumeral", "scapulothoracic", "acromioclavicular", "sternoclavicular"],
        [1, 2, 3],
        1922447.0594910611,
        [(0, 5.512546219515222), (1, 5.512549776544782), (2, 5.512606226422835), (-1, -0.44703921745960906)],
    ),
    "Oki et al.": (
        354,
        ["frontal plane elevation", "sagittal plane elevation", "horizontal flexion"],
        ["scapulothoracic", "sternoclavicular"],
        [1, 2, 3],
        2343.222633228853,
        [(0, -23.5715), (100, 23.698003331400965), (200, 15.424283835508106), (-1, 31.7351)],
    ),
    "Teece et al.": (
        39,
        ["scapular plane elevation"],
        ["acromioclavicular"],
        [1, 2, 3],
        1061.3874465806718,
        [(0, 53.208998323357854), (10, 55.34423815737802), (22, 12.65697336292887), (-1, 21.697663050779525)],
    ),
    "Yoshida et al.": (
        84,
        ["sagittal plane elevation"],
        ["glenohumeral", "scapulothoracic"],
        [1, 2, 3],
        551.3161398751674,
        [(0, 44.304576869953884), (40, -24.893753727658684), (65, 34.51266), (-1, 19.2415854)],
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

    if article_name == "Kozono et al.":
        # Skip this test because the thorax is indirect but once we decide which one to use we can remove this line
        return

    print_data(data, random_checks)
    assert data.shape[0] == expected_shape

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
        "Sahara et al." "Sugi et al.",
        "Teece et al.",
        "Yoshida et al.",
    ]
    assert articles == experted_articles

    assert confident_values.shape[0] == 812040


def print_data(data, random_checks):
    print("\n")
    print("Shape:", data.shape)
    print("Humeral motions:", data["humeral_motion"].unique())
    print("Joints:", data["joint"].unique())
    print("Degrees of freedom:", data["degree_of_freedom"].unique())
    print("Total value:", data["value"].sum())
    print("Random checks:")
    for idx, value in random_checks:
        print(f"Data {idx}: {data['value'].iloc[idx]}")
        print(f"Check {idx}: {value}")
    print("")
