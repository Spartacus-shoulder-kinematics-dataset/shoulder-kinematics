import streamlit as st
from spartacus import DataPlanchePlotting, DataFrameInterface, import_data

st.set_page_config(
    page_title="Spartacus",
    page_icon="💪",
    layout="wide",  # Use 'wide' layout
    initial_sidebar_state="expanded",
)

col1, col2, col3 = st.columns([1, 6, 1])  # Center column (col2) is much wider


# Load and prepare the data
@st.cache_data
def load_data():
    df = import_data(correction=True)
    df = df[df["unit"] == "rad"]
    return df


df = load_data()
dfi = DataFrameInterface(df)

# Streamlit app layout

# Dropdown to select humeral motion
selected_humeral_motion = st.sidebar.selectbox(
    "Select Humeral Motion:",
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
    "Select Joints to Display:",
    options=["glenohumeral", "scapulothoracic", "acromioclavicular", "sternoclavicular"],
    default=["glenohumeral", "scapulothoracic", "acromioclavicular", "sternoclavicular"],  # Default to all joints
)

# Radio buttons to select experimental mean
experimental_mean_option = st.sidebar.multiselect(
    "Select Experimental Mean:",
    options=["Pins", "Biplane X-ray fluoroscopy", "Single-plane X-ray fluoroscopy", "MRI", "4DCT"],
    default=["Pins", "Biplane X-ray fluoroscopy", "Single-plane X-ray fluoroscopy", "MRI", "4DCT"],
)

# Selectbox to choose active or passive
invivo_option = st.sidebar.multiselect(
    "Select Active or Passive:", options=["In Vivo", "Ex Vivo"], default=["In Vivo", "Ex Vivo"]
)  # Default to "Active"


# Selectbox to choose whether Thorax is Global or Local
thorax_option = st.sidebar.multiselect(
    "Select Thorax Coordinate Systems:", options=["Global", "Local"], default=["Global", "Local"]  # Default to "Global"
)

# Selectbox to choose posture
posture_option = st.sidebar.multiselect(
    "Select Postures:", options=["Standing", "Sitting"], default=["Standing", "Sitting"]
)  # Default to "Global"

# Selectbox to choose type of movement
movement_type_option = st.sidebar.multiselect(
    "Select Types of Movement:",
    options=["Dynamic", "Quasi-static"],
    default=["Dynamic", "Quasi-static"],  # Default to "Dynamic"
)

# Selectbox to choose active or passive
active_option = st.sidebar.multiselect(
    "Select Active or Passive:", options=["Active", "Passive"], default=["Active", "Passive"]
)  # Default to "Active"

# Compliance Slider
total_compliance = st.sidebar.slider("ISB compliance from 0 (Not at all) to 6 (Entirely Compliant)", 0, 6, 0)
st.write("Datasets that only have a ISB compliance over ", total_compliance, " are displayed.")

# Filter the DataFrame based on the selected humeral motion and joints
subdf = df[df["humeral_motion"] == selected_humeral_motion]
subdf = subdf[subdf["joint"].isin(selected_joints)]

# Apply the filter for in vivo, ex vivo, or both
in_vivo_bool = [option == "In Vivo" for option in invivo_option]
subdf = subdf[subdf["in_vivo"].isin(in_vivo_bool)]

# Apply the filter for Thorax Global or Local
thorax_option = [option == "Global" for option in thorax_option]
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

# Initialize the DataPlanchePlotting object with selected options
dfi_filtered = DataFrameInterface(subdf)
plt = DataPlanchePlotting(dfi_filtered, restrict_to_joints=selected_joints)
plt.plot()
plt.update_style_streamlit()

# Display the plot using Streamlit
with col2:
    st.plotly_chart(plt.fig, use_container_width=True, theme=None)
