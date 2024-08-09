from spartacus import Spartacus
import os


def main():
    sp = Spartacus.load(check_and_import=False)
    sp.dataframe.to_csv("merged_dataframe.csv")


if __name__ == "__main__":
    main()
