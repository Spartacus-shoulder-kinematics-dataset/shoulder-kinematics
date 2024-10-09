"""
How to launch the app, write in the command line:
$ streamlit run app_streamlit.py
"""

import streamlit as st

from spartacus import DataPlanchePlotting, DataFrameInterface, import_data
import pandas as pd


def set_page_config():
    st.set_page_config(
        page_title="Spartacus",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data
def load_data():
    df = import_data(correction=True)
    return df


def create_side_bar_components():
    st.sidebar.header("Options")

    selected_metric = st.sidebar.selectbox(
        "Metric:",
        options=["rotations", "translations"],
        index=0,  # Default to the first option
    )
    selected_metric_map = {"rotations": "rad", "translations": "mm"}
    selected_metric = selected_metric_map[selected_metric]

    selected_humeral_motion = st.sidebar.selectbox(
        "Humeral Motion:",
        options=[
            "frontal plane elevation",
            "scapular plane elevation",
            "sagittal plane elevation",
            "internal-external rotation 0 degree-abducted",
            "internal-external rotation 90 degree-abducted",
            "horizontal flexion",
        ],
        index=1,  # Default to the first option
    )

    # Checklist to select joints
    selected_joints = st.sidebar.multiselect(
        "Shoulder joints:",
        options=["glenohumeral", "scapulothoracic", "acromioclavicular", "sternoclavicular"],
        default=["glenohumeral", "scapulothoracic", "acromioclavicular", "sternoclavicular"],  # Default to all joints
    )

    # Radio buttons to select experimental mean
    experimental_mean_option = st.sidebar.multiselect(
        "Experimental Means:",
        options=["Pins", "Biplane X-ray fluoroscopy", "Single-plane X-ray fluoroscopy", "MRI", "4DCT"],
        default=["Pins", "Biplane X-ray fluoroscopy", "Single-plane X-ray fluoroscopy", "MRI", "4DCT"],
    )

    # Selectbox to choose active or passive
    invivo_option = st.sidebar.multiselect(
        "Subjects:", options=["In Vivo", "Ex Vivo"], default=["In Vivo", "Ex Vivo"]
    )  # Default to "Active"

    # Selectbox to choose whether Thorax is Global or Local
    thorax_options = st.sidebar.multiselect(
        "Thorax Coordinate Systems:", options=["Global", "Local"], default=["Global", "Local"]  # Default to "Global"
    )

    # Selectbox to choose posture
    posture_option = st.sidebar.multiselect(
        "Postures:", options=["Standing", "Sitting"], default=["Standing", "Sitting"]
    )  # Default to "Global"

    # Selectbox to choose type of movement
    movement_type_option = st.sidebar.multiselect(
        "Types of Movement:",
        options=["Dynamic", "Quasi-static"],
        default=["Dynamic", "Quasi-static"],  # Default to "Dynamic"
    )

    # Selectbox to choose active or passive
    active_option = st.sidebar.multiselect(
        "Active or Passive Movements:", options=["Active", "Passive"], default=["Active", "Passive"]
    )  # Default to "Active"

    # Compliance Slider
    total_compliance = st.sidebar.slider("ISB compliance from 0 (Not at all) to 6 (Entirely Compliant)", 0, 6, 0)

    with st.sidebar.expander("Option Explanations"):
        st.write(
            """
            **Humeral Motions**: 
            Select the type of humeral motion to analyze. This includes various movements 
            such as elevation in different planes, which are recorded across the datasets.

            **Shoulder Joints**: 
            Choose which joints' data to include in the visualization. You can select 
            multiple joints to compare their behaviors during the selected motion.

            **Data Type (In Vivo/Ex Vivo)**: 
            Filter the data based on whether it was collected in vivo (within a living 
            organism) or ex vivo (with a cadaver).

            **Active/Passive Movement**: 
            Display data from active movements (muscle-driven) or passive movements (externally driven).

            **Compliance Range**: 
            The compliance score quantifies how closely the biomechanical model adheres 
            to established standards (ISB - International Society of Biomechanics). It is 
            calculated based on the compliance of segments and joints. 

            Use the slider to filter the data based on the total compliance score, which 
            ranges from 0 (no adherence) to 6 (sticks to the standard).

            - **Segment Compliance**: Assesses the alignment and origin of biomechanical 
              coordinate systems with ISB standards.
            - **Joint Compliance**: Evaluates joint-specific factors like Euler sequence 
              and translation frames relative to ISB guidelines.
            - **Total Compliance**: A composite score derived from segment and joint 
              compliance metrics, indicating the overall adherence to biomechanical 
              standards. Differ if Rotations or Translations.

            Rotation compliance is calculated by adding up the compliance scores 
            from both the proximal (parent) and distal (child) segments, 
            focusing on their ISB orientation and axis construction (c1, c2), 
            as well as the joint's Euler sequence (if ISB) and thoracohumeral angle, 
            if also based on a Euler sequence or not (c4, c6).

            """
        )
    st.sidebar.header("Cite this work")
    st.sidebar.write(
        """
            **Moissenet, F., Puchaud, P., Naaïm, A., Holzer, N., & Begon, M. (2024).** *Spartacus-shoulder-kinematics-dataset/shoulder-kinematics congress (0.1.0).* Zenodo.[https://doi.org/10.5281/zenodo.11455521](https://doi.org/10.5281/zenodo.11455521)
            """
    )
    return (
        selected_metric,
        selected_humeral_motion,
        selected_joints,
        experimental_mean_option,
        invivo_option,
        thorax_options,
        posture_option,
        movement_type_option,
        active_option,
        total_compliance,
    )


def filter_dataframe(
    df,
    selected_metric,
    selected_humeral_motion,
    selected_joints,
    experimental_mean_option,
    invivo_option,
    thorax_options,
    posture_option,
    movement_type_option,
    active_option,
    total_compliance,
):
    # Filter if translation or rotation data only
    sub_df = df[df["unit"] == selected_metric]

    # Filter the DataFrame based on the selected humeral motion and joints
    subdf = sub_df[sub_df["humeral_motion"] == selected_humeral_motion]
    subdf = subdf[subdf["joint"].isin(selected_joints)]

    # Apply the filter for in vivo, ex vivo, or both
    in_vivo_bool = [option == "In Vivo" for option in invivo_option]
    subdf = subdf[subdf["in_vivo"].isin(in_vivo_bool)]

    # Apply the filter for Thorax Global or Local
    thorax_option = [option == "Global" for option in thorax_options]
    subdf = subdf[subdf["thorax_is_global"].isin(thorax_option)]

    # Apply the filter for posture
    subdf = subdf[subdf["posture"].isin([p.lower() for p in posture_option])]
    # Apply the filter for type of movement
    subdf = subdf[subdf["type_of_movement"].isin([m.lower() for m in movement_type_option])]

    # Apply the filter for active or passive
    active_bool = [option == "Active" for option in active_option]
    subdf = subdf[subdf["active"].isin(active_bool)]

    # Apply the filter for experimental mean
    xp_map = {
        "Pins": "intra cortical pins",
        "Biplane X-ray fluoroscopy": "biplane x-ray fluoroscopy",
        "Single-plane X-ray fluoroscopy": "single-plane x-ray fluoroscopy",
        "MRI": "MRI",
        "4DCT": "4DCT",
    }
    subdf = subdf[subdf["experimental_mean"].isin([xp_map[e] for e in experimental_mean_option])]

    # Filter Compliance
    subdf = subdf[subdf["total_compliance"] >= total_compliance]

    return subdf


def plot(df, selected_joints):
    # Initialize the DataPlanchePlotting object with selected options
    dfi_filtered = DataFrameInterface(df)
    plt = DataPlanchePlotting(dfi_filtered, restrict_to_joints=selected_joints)
    plt.plot()
    plt.update_style_streamlit()

    return plt


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


# Starting Here !
set_page_config()

if "df_perso" not in st.session_state:
    st.session_state.df_perso = None

# Center column (col2) is much wider
col1, col2, col3 = st.columns([0.2, 6, 1])
# load the dataframe
df = load_data()
csv = convert_df(df)

(
    selected_metric,
    selected_humeral_motion,
    selected_joints,
    experimental_mean_option,
    invivo_option,
    thorax_options,
    posture_option,
    movement_type_option,
    active_option,
    total_compliance,
) = create_side_bar_components()

if st.session_state.df_perso is not None:
    df = pd.concat([st.session_state.df_perso, df])

dfi = DataFrameInterface(df)

df_filtered = filter_dataframe(
    df,
    selected_metric,
    selected_humeral_motion,
    selected_joints,
    experimental_mean_option,
    invivo_option,
    thorax_options,
    posture_option,
    movement_type_option,
    active_option,
    total_compliance,
)

csv_filtered = convert_df(df_filtered)
plt = plot(df_filtered, selected_joints)

# Display the plot using Streamlit
with col2:
    st.plotly_chart(plt.fig, use_container_width=True, theme=None)
    st.write("Datasets that only have a ISB compliance over ", total_compliance, " are displayed.")

with col3:
    st.image(
        "docs/logo_only.png",
        caption="Spartacus Dataset. \n Moissenet, F., Puchaud, P., Naaïm, A., Holzer, N., & Begon, M. (2024)",
    )
    st.download_button(
        label="Download Spartacus \n as CSV",
        data=csv,
        file_name="spartacus.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download my selection \n as CSV",
        data=csv_filtered,
        file_name="spartacus_selection.csv",
        mime="text/csv",
    )


# File uploader widget
st.header("Upload Your Data")
df_perso = st.file_uploader("Choose a file", type=["csv", "txt"])
# Expander to explain the expected CSV format
with st.expander("See the expected .csv format"):
    st.write(
        """
        The CSV file should have the following columns:
        
        - **article**: The reference or study associated with the data.
        - **unit**: The unit of measurement (e.g., 'rad' for radians).
        - **joint**: The joint related to the data (e.g., glenohumeral, scapulothoracic).
        - **humeral_motion**: The type of motion being analyzed (e.g., frontal plane elevation).
        - **total_compliance**: A numeric value representing the compliance score (e.g., 6 if you want it always displayed).
        - **humerothoracic_angle**: The angle measured between the humerus and thorax (i.e., abscissa).
        - **value**: The recorded value related to the motion (i.e., ordinate).
        - **degree_of_freedom**: The degree of freedom of the value (1, 2, or 3).
        - **in_vivo**: Boolean indicating if the data was collected in vivo (True/False).
        - **experimental_mean**: The mean derived from experiments, such as CT-scan data.
        - **type_of_movement**: Describes the movement type (e.g., dynamic, quasi-static).
        - **active**: Indicates if the movement was active or passive (True/False).
        - **posture**: The posture during data collection (e.g., standing, sitting).
        - **thorax_is_global**: Boolean indicating if the thorax was considered from the global coordinate system.

        **Example .csv Format**:
        ```
        article,unit,joint,humeral_motion,total_compliance,humerothoracic_angle,value,degree_of_freedom,in_vivo,experimental_mean,type_of_movement,active,posture,thorax_is_global
        Ours,rad,glenohumeral,frontal plane elevation,6.0,45.0,45.3,1,True,intra cortical pins,dynamic,True,standing,False
        Ours,rad,glenohumeral,frontal plane elevation,6.0,120,50,1,True,intra cortical pins,dynamic,True,standing,False
        Ours,rad,scapulothoracic,sagittal plane elevation,3.0,30.5,30.7,2,False,biplane x-ray fluoroscopy,quasi-static,False,sitting,True
        Ours,rad,scapulothoracic,sagittal plane elevation,3.0,120,30.7,2,False,biplane x-ray fluoroscopy,quasi-static,False,sitting,True
        ```
        """
    )

if df_perso is not None:
    st.session_state.df_perso = pd.read_csv(df_perso, delimiter=",")
    st.success("File uploaded and processed successfully!")
