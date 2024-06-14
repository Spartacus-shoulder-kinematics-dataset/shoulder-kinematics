import collections

nested_dict = lambda: collections.defaultdict(nested_dict)
DEVIATION_COEFF = nested_dict()
DEVIATION_COEFF["rotation"]["label"] = 0.9
DEVIATION_COEFF["rotation"]["sens"] = 0.9
DEVIATION_COEFF["rotation"]["origin"] = 0.9
DEVIATION_COEFF["rotation"]["direction"] = 0.5

DEVIATION_COEFF["displacement"]["label"] = 0.9
DEVIATION_COEFF["displacement"]["sens"] = 0.9
DEVIATION_COEFF["displacement"]["origin"] = 0.5
DEVIATION_COEFF["displacement"]["direction"] = 0.5
