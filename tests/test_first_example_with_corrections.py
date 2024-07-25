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
    "Bourne et al.": (
        2550,
        ["frontal elevation", "horizontal flexion"],
        ["scapulothoracic"],
        [1, 2, 3],
        32987.06725953286,
        [(0, -16.366300000000003), (1001, 21.925651818083278), (2000, -38.92291527743668), (-1, 17.319848672019763)],
    ),
    "Chu et al.": (
        96,
        ["frontal elevation", "scapular elevation", "internal-external rotation 90 degree-abducted"],
        ["scapulothoracic"],
        [1, 2, 3],
        -554.7492646716876,
        [(0, 20.832700000000003), (30, -2.4029921148049445), (60, -8.436784721537704), (-1, -4.961429503563281)],
    ),
    "Fung et al.": (
        621,
        ["frontal elevation", "scapular elevation", "sagittal elevation"],
        ["scapulothoracic"],
        [1, 2, 3],
        3676.4321999999997,
        [(0, 36.84060000000001), (30, 29.6062), (60, 18.136000000000003), (-1, 9.294200000000002)],
    ),
    "Kijima et al.": (
        48,
        ["scapular elevation"],
        ["glenohumeral", "scapulothoracic"],
        [1, 2, 3],
        306.06436262040665,
        [(0, np.nan), (1, np.nan), (2, np.nan), (-1, 7.421436415016586)],
    ),
    "Begon et al.": (
        19296,
        [
            "frontal elevation",
            "sagittal elevation",
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
        -12714.85146291502,
        [(0, 162.73707354679948), (1001, -47.33668895380996), (2000, -42.92927676074317), (-1, 12.282290646030903)],
    ),
    "Henninger et al.": (
        80862,
        [
            "frontal elevation",
            "scapular elevation",
            "sagittal elevation",
            "internal-external rotation 0 degree-abducted",
            "internal-external rotation 90 degree-abducted",
        ],
        ["glenohumeral", "scapulothoracic"],
        [1, 2, 3],
        1753489.2980004128,
        [(0, 45.9866459013967), (1001, 76.9873557465161), (40001, -8.61904452693329), (-1, 18.710579311164597)],
    ),
    "Kozono et al.": (
        30,
        ["internal-external rotation 0 degree-abducted"],
        ["glenohumeral"],
        [1, 2, 3],
        0,
        [(0, np.nan), (1, np.nan), (2, np.nan), (-1, np.nan)],
    ),
    "Ludewig et al.": (
        684,
        ["frontal elevation", "scapular elevation", "sagittal elevation"],
        ["glenohumeral", "scapulothoracic", "acromioclavicular", "sternoclavicular"],
        [1, 2, 3],
        9001.038459374906,
        [(0, 137.332189001885), (1, 138.62633045180257), (2, 139.82856070895113), (-1, 25.0)],
    ),
    "Matsumura et al.": (
        99,
        ["frontal elevation", "scapular elevation", "sagittal elevation"],
        ["scapulothoracic"],
        [1, 2, 3],
        -558.4569560038939,
        [(0, -23.068), (20, 32.595395315826835), (60, -0.8599417044686809), (-1, 11.970999999999998)],
    ),
    "Matsuki et al.": (
        1836,  #     288
        ["scapular elevation"],
        ["scapulothoracic", "glenohumeral", "sternoclavicular"],
        [1, 2, 3],
        -18922.510132213356,
        [(0, 0.7331976380416924), (1, 1.3163822061108446), (2, 2.009989683892475), (-1, np.nan)],
    ),
    "Moissenet et al.": (
        705264,
        [
            "frontal elevation",
            "sagittal elevation",
            "horizontal flexion",
            "internal-external rotation 0 degree-abducted",
        ],
        ["glenohumeral", "scapulothoracic", "acromioclavicular", "sternoclavicular"],
        [1, 2, 3],
        8884019.476285355,
        [(0, -174.48745378048477), (1, -174.48745022345523), (2, -174.48739377357717), (-1, -0.447039217459609)],
    ),
    "Oki et al.": (
        354,
        ["frontal elevation", "sagittal elevation", "horizontal flexion"],
        ["scapulothoracic", "sternoclavicular"],
        [1, 2, 3],
        2343.2226332288533,
        [(0, -23.571499999999997), (100, 23.698003331400965), (200, 15.424283835508106), (-1, 31.7351)],
    ),
    "Teece et al.": (
        39,
        ["scapular elevation"],
        ["acromioclavicular"],
        [1, 2, 3],
        1070.5063803329367,  # Repeating total value here as a placeholder
        [(0, 53.208998323357854), (10, 55.34423815737802), (22, 12.65697336292887), (-1, 21.697663050779525)],
        # Random checks
    ),
    "Yoshida et al.": (
        84,
        ["sagittal elevation"],
        ["glenohumeral", "scapulothoracic"],
        [1, 2, 3],
        1445.1089898448326,
        [(0, -44.304576869953905), (40, -16.641814272341282), (65, 34.51266), (-1, 19.2415854)],
    ),
    # Add other articles here in the same format
}
transformed_data_article = [[name] + list(values) for name, values in articles_data.items()]


spartacus = TestUtils.spartacus_folder()
module = TestUtils.load_module(spartacus + "/examples/first_example.py")
confident_values = module.main()


# This line parameterizes the test function below
@pytest.mark.parametrize(
    "article_name,expected_shape,humeral_motions,joints,dofs,total_value,random_checks", transformed_data_article
)
def test_article_data(article_name, expected_shape, humeral_motions, joints, dofs, total_value, random_checks):
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
    expected_articles = [
        "Begon et al.",
        "Bourne et al.",
        "Chu et al.",
        "Fung et al.",
        "Henninger et al.",
        "Karduna et al.",
        "Kijima et al.",
        # "Kozono et al.", todo: to be put again when thorax is decided
        "Ludewig et al.",
        "Matsuki et al.",
        "Matsumura et al.",
        "Moissenet et al.",
        "Oki et al.",
        "Teece et al.",
        "Yoshida et al.",
    ]
    assert articles == expected_articles

    assert len(articles) == 14

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
