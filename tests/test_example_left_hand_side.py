from .utils import TestUtils

spartacus = TestUtils.spartacus_folder()
module = TestUtils.load_module(spartacus + "/examples/playground/left_and_right_side_rotation_matrices.py")
confident_values = module.main()


def test_left_handed_rotation_matrix():
    module.main()
