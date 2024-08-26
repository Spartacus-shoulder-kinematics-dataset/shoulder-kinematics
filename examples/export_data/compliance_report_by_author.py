import numpy as np
from spartacus import Spartacus


def main():
    sp = Spartacus.load(check_and_import=False)
    df = sp.compliance()

    return df.replace(np.nan, "-")


if __name__ == "__main__":
    df = main()
    df = df.replace(True, 1)
    df = df.replace(False, 0)
    df.to_csv("deviation_table.csv")
    print(df.values)
