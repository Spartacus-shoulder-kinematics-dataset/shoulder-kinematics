import seaborn as sns
import numpy as np

BIOMECHANICAL_DOF_LEGEND = {
    "glenohumeral": ("Plane of elevation", "Elevation(-)/Depression(+)", "Internal(+)/external(-) rotation"),
    "scapulothoracic": (
        "Protraction(+)/retraction(-)",
        "Medial(+)/lateral(-) rotation",
        "Posterior(+)/anterior(-) tilt",
    ),
    "acromioclavicular": (
        "Protraction(+)/retraction(-)",
        "Medial(+)/lateral(-) rotation",
        "Posterior(+)/anterior(-) tilt",
    ),
    "sternoclavicular": (
        "Protraction(+)/retraction(-)",
        "Depression(+)/elevation(-)",
        "Backwards(+)/forward(-) rotation",
    ),
    "thoracohumeral": ("Plane of elevation", "Elevation", "Internal(+)/external(-) rotation"),
}

TRANSLATIONAL_BIOMECHANICAL_DOF_LEGEND = {
    "glenohumeral": ("postero(-)/anterior(+)", "infero(-)/superior(+)", "medio(-)/lateral(+)"),
    "scapulothoracic": ("postero(-)/anterior(+)", "infero(-)/superior(+)", "medio(-)/lateral(+)"),
    "acromioclavicular": ("postero(-)/anterior(+)", "infero(-)/superior(+)", "medio(-)/lateral(+)"),
    "sternoclavicular": ("postero(-)/anterior(+)", "infero(-)/superior(+)", "medio(-)/lateral(+)"),
    "thoracohumeral": ("postero(-)/anterior(+)", "infero(-)/superior(+)", "medio(-)/lateral(+)"),
}

JOINT_ROW_COL_INDEX = {
    "glenohumeral": ((0, 0), (0, 1), (0, 2)),
    "scapulothoracic": ((1, 0), (1, 1), (1, 2)),
    "acromioclavicular": ((2, 0), (2, 1), (2, 2)),
    "sternoclavicular": ((3, 0), (3, 1), (3, 2)),
}

MVT_ROW_COL_INDEX = {
    "frontal plane elevation": ((0, 0), (0, 1), (0, 2)),
    "scapular plane elevation": ((1, 0), (1, 1), (1, 2)),
    "sagittal plane elevation": ((2, 0), (2, 1), (2, 2)),
    "internal-external rotation 0 degree-abducted": ((3, 0), (3, 1), (3, 2)),
    "internal-external rotation 90 degree-abducted": ((4, 0), (4, 1), (4, 2)),
}


def rgb_to_hex(rgb):
    # Scale the RGB values from [0, 1] to [0, 255]
    scaled_rgb = tuple(int(val * 255) for val in rgb)

    # Convert to hexadecimal format
    hex_color = "#{:02x}{:02x}{:02x}".format(*scaled_rgb)

    return hex_color


def author_colors_constant():
    # Create the main "icefire" palette as a NumPy array
    # main_palette = np.array(sns.color_palette("icefire", n_colors=21))
    main_palette = np.array(sns.color_palette("icefire", n_colors=18))
    main_palette = np.concatenate(
        (
            np.array([[170 / 255, 255 / 255, 127 / 255]]),
            np.array([[127 / 255, 255 / 255, 148 / 255]]),
            np.array([[127 / 255, 255 / 255, 212 / 255]]),
            main_palette,
        )
    )
    # Create the first half of the new palette by linearly interpolating colors
    start_index = 0
    end_index = 10
    new_x = np.linspace(start_index, end_index, num=15)
    interpolated_colors_ice = np.empty((len(new_x), 3))
    for i in range(3):
        interpolated_colors_ice[:, i] = np.interp(new_x, np.arange(21), main_palette[:, i])

    # # Add a green sine
    # period = 2 * np.pi / 21 * 4
    # interpolated_colors_ice[::2, 0] += np.abs(0.15 * np.sin(period * new_x[0::2] + np.pi / 4))
    # interpolated_colors_ice[::2, 1] += 0.12 * np.sin(period * new_x[0::2] + np.pi / 4)

    # Create the second half of the new palette by linearly interpolating colors
    start_index = 15
    end_index = 20
    new_x = np.linspace(start_index, end_index, num=6)
    interpolated_colors_fire = np.empty((len(new_x), 3))
    for i in range(3):
        interpolated_colors_fire[:, i] = np.interp(new_x, np.arange(21), main_palette[:, i])

    # Concatenate the two halves of the new palette
    palette = np.concatenate((interpolated_colors_ice, interpolated_colors_fire), axis=0)

    # import matplotlib.pyplot as plt
    # sns.palplot(palette)
    # plt.show()

    # Convert the palette back to a list of tuples
    palette = [tuple(color) for color in palette]

    # return {
    #     # In vivo
    #     "Begon et al.": palette[0],
    #     "Bourne et al.": palette[1],
    #     "Chu et al.": palette[2],
    #     "Henninger et al.": palette[3],
    #     "Karduna et al.": palette[4],
    #     "Kijima et al.": palette[5],
    #     "Kim et al.": palette[6],
    #     "Kozono et al.": palette[7],
    #     "Ludewig et al.": palette[8],
    #     "Malberg et al.": palette[9],
    #     "Matsuki et al.": palette[10],
    #     "Nishinaka et al.": palette[11],
    #     "Sahara et al.": palette[12],
    #     "Sugi et al.": palette[13],
    #     "Yoshida et al.": palette[14],
    #     # ex vivo
    #     "Fung et al.": palette[20],
    #     "Gutierrez Delgado et al.": palette[19],
    #     "Matsumura et al.": palette[18],
    #     "Moissenet et al.": palette[17],
    #     "Oki et al.": palette[16],
    #     "Teece et al.": palette[15],
    # }
    # return {
    #     # In vivo
    #     "Begon et al.": "#AAFF7F",
    #     "Bourne et al.": "#7FFF94",
    #     "Chu et al.": "#7FFFD4",
    #     "Henninger et al.": "#00FFD1",
    #     "Karduna et al.": "#00D4FF",
    #     "Kijima et al.": "#00B0FF",
    #     "Kim et al.": "#0091FF",
    #     "Kozono et al.": "#0071FF",
    #     "Ludewig et al.": "#005BFF",
    #     "Malberg et al.": "#0040FF",
    #     "Matsuki et al.": "#002AFF",
    #     "Nishinaka et al.": "#1A00FF",
    #     "Sahara et al.": "#3800FF",
    #     "Sugi et al.": "#5600FF",
    #     "Yoshida et al.": "#7300FF",
    #     # Ex vivo
    #     "Fung et al.": "#FFAA7F",
    #     "Gutierrez Delgado et al.": "#FF7F50",
    #     "Matsumura et al.": "#FF6A00",
    #     "Moissenet et al.": "#FF4500",
    #     "Oki et al.": "#FF2100",
    #     "Teece et al.": "#B20000",
    # }
    return {
        # In vivo
        "Begon et al.": (170, 255, 127),
        "Bourne et al.": (127, 255, 148),
        "Chu et al.": (127, 255, 212),
        "Henninger et al.": (0, 255, 209),
        "Karduna et al.": (0, 212, 255),
        "Kijima et al.": (0, 176, 255),
        "Kim et al.": (0, 145, 255),
        "Kozono et al.": (0, 113, 255),
        "Ludewig et al.": (0, 91, 255),
        "Malberg et al.": (0, 64, 255),
        "Matsuki et al.": (0, 42, 255),
        "Nishinaka et al.": (26, 0, 255),
        "Sahara et al.": (56, 0, 255),
        "Sugi et al.": (86, 0, 255),
        "Yoshida et al.": (115, 0, 255),
        # Ex vivo
        "Fung et al.": (255, 170, 127),
        "Gutierrez Delgado et al.": (255, 127, 80),
        "Matsumura et al.": (255, 106, 0),
        "Moissenet et al.": (255, 69, 0),
        "Oki et al.": (255, 33, 0),
        "Teece et al.": (178, 0, 0),
    }


AUTHORS_COLORS = author_colors_constant()


AUTHOR_DISPLAYED_STUDY = {
    # In vivo
    "Begon et al.": "#1 Begon et al.",
    "Bourne et al.": "#2 Bourne et al.",
    "Chu et al.": "#3 Chu et al.",
    "Henninger et al.": "#6a Henninger et al.",
    "Henninger et al. 6b AC": "#6b Henninger et al.",
    "Henninger et al. 6c GC": "#6c Henninger et al.",
    "Karduna et al.": "#7 Karduna et al.",
    "Kijima et al.": "#8 Kijima et al.",
    "Kim et al.": "#9 Kim et al.",
    "Kozono et al.": "#10 Kozono et al.",
    "Ludewig et al.": "#11 Ludewig et al.",
    "Malberg et al.": "#12 Malberg et al.",
    "Matsuki et al.": "#13 Matsuki et al.",
    "Nishinaka et al.": "#16 Nishinaka et al.",
    "Sahara et al.": "#18 Sahara et al.",
    "Sugi et al.": "#19 Sugi et al.",
    "Yoshida et al.": "#21 Yoshida et al.",
    # ex vivo
    "Fung et al.": "#4 Fung et al.",
    "Gutierrez Delgado et al.": "#5 Gutierrez Delgado et al.",
    "Matsumura et al.": "#14 Matsumura et al.",
    "Moissenet et al.": "#15 Moissenet et al.",
    "Oki et al.": "#17 Oki et al.",
    "Teece et al.": "#20 Teece et al.",
}
