from pandas import DataFrame

from spartacus import DataFolder, Spartacus, DataFrameInterface, DataPlanchePlotting


def main():
    # spartacus_dataset = load_subdataset(name=DataFolder.BEGON_2014, shoulder=1, mvt="sagittal plane elevation")
    # spartacus_dataset = load_subdataset(name=DataFolder.BEGON_2014)
    spartacus_dataset = Spartacus.load(
        datasets=[DataFolder.OKI_2012, DataFolder.BOURNE_2003, DataFolder.MOISSENET],
        mvt="horizontal flexion",
        # shoulder=[i for i in range(1, 8)],
    )
    print(spartacus_dataset.confident_data_values)
    return spartacus_dataset.corrected_confident_data_values
    # if ones wants to return the uncompensated data:
    # return spartacus_dataset.confident_data_values


def plot_mvt(df: DataFrame):

    humeral_motions = df["humeral_motion"].unique()

    for mvt in humeral_motions:
        sub_df = df[df["humeral_motion"] == mvt]
        dfi = DataFrameInterface(sub_df).rotational_interface()
        plt = DataPlanchePlotting(dfi, restrict_to_joints=dfi.df["joint"].unique())
        plt.plot()
        plt.update_style()
        plt.show()


if __name__ == "__main__":
    data = main()
    plot_mvt(data)
