import seaborn as sns


BIOMECHANICAL_DOF_LEGEND = {
    "glenohumeral": ("Plane of elevation", "Elevation", "Internal(+)/external(-) rotation"),
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


def rgb_to_hex(rgb):
    # Scale the RGB values from [0, 1] to [0, 255]
    scaled_rgb = tuple(int(val * 255) for val in rgb)

    # Convert to hexadecimal format
    hex_color = "#{:02x}{:02x}{:02x}".format(*scaled_rgb)

    return hex_color


def author_colors_constant():
    # palette = sns.color_palette(cc.glasbey, n_colors=20)
    # palette = cc.linear_bmw_5_95_c86
    # convert [0, 1] to hexa
    # palette = [rgb_to_hex(color) for color in palette]
    # palette = sns.color_palette(palette, n_colors=20)
    palette = sns.color_palette("icefire", n_colors=21)
    # palette = sns.diverging_palette(220, 0, l=50, s=80, n=20, center="light")
    # print(palette)
    # display the palette

    # import matplotlib.pyplot as plt
    #
    # sns.palplot(palette)
    # plt.show()
    # palette_c = sns.color_palette(cc.glasbey_cool, n_colors=20)

    return {
        # In vivo
        "Begon et al.": palette[0],
        "Bourne et al.": palette[1],
        "Chu et al.": palette[2],
        "Henninger et al.": palette[3],
        "Kijima et al.": palette[4],
        "Kim et al.": palette[5],
        "Kozono et al.": palette[6],
        "Ludewig et al.": palette[7],
        "Malberg et al.": palette[8],
        "Matsuki et al.": palette[9],
        "Nishinaka et al.": palette[10],
        "Sahara et al.": palette[11],
        "Sugi et al.": palette[12],
        "Yoshida et al.": palette[13],
        # ex vivo
        "Fung et al.": palette[19],
        "Gutierrez Delgado et al.": palette[18],
        "Matsumura et al.": palette[17],
        "Moissenet et al.": palette[16],
        "Oki et al.": palette[15],
        "Teece et al.": palette[14],
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
