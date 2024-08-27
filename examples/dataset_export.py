from spartacus import Spartacus


def main():
    sp = Spartacus.load(unify=False)
    sp.dataframe.to_csv("merged_dataframe.csv")


if __name__ == "__main__":
    main()
