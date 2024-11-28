from pandas import DataFrame

from spartacus import DataFolder, Spartacus, DataFrameInterface, DataPlanchePlotting


def main(compensated=True):
    """
    Load the Spartacus dataset and plot the scapulothoracic

    Parameters
    ----------
    compensated: bool
        If True, return the compensated data, otherwise return the uncompensated data
    """
    # spartacus_dataset = load_subdataset(name=DataFolder.BEGON_2014, shoulder=1, mvt="sagittal plane elevation")
    # spartacus_dataset = load_subdataset(name=DataFolder.BEGON_2014)
    spartacus_dataset = Spartacus.load(
        # datasets=[DataFolder.OKI_2012, DataFolder.BOURNE_2003, DataFolder.MOISSENET],
        mvt=["sagittal plane elevation", "frontal plane elevation", "scapular plane elevation"],
        joints=["scapulothoracic"],
        # shoulder=[i for i in range(1, 8)],
    )
    print(spartacus_dataset.confident_data_values)
    if compensated:
        return spartacus_dataset.corrected_confident_data_values
    else:
        return spartacus_dataset.confident_data_values


def plot_mvt(df: DataFrame, compensated=True):

    humeral_motions = df["humeral_motion"].unique()

    for mvt in humeral_motions:
        sub_df = df[df["humeral_motion"] == mvt]
        dfi = DataFrameInterface(sub_df).rotational_interface()
        plt = DataPlanchePlotting(dfi, restrict_to_joints=dfi.df["joint"].unique())
        plt.plot()
        plt.update_style()
        plt.show()

        extra_suffix = "_compensated" if compensated else "_uncompensated"
        plt.fig.write_image(f"{mvt}_ST{extra_suffix}.svg")


if __name__ == "__main__":
    data = main(compensated=True)
    plot_mvt(data, compensated=True)
    data = main(compensated=False)
    plot_mvt(data, compensated=False)
