
<p align="center" width="100%">
<img src="docs/logo.png" alt="Spartacus" style="width: 50%; min-width: 300px; display: block; margin: auto;">
</p>

# Introduction
Gathering all the literature on shoulder kinematics, and scapulo-humeral rhythm
With this repository, We aim to gather all the literature on shoulder kinematics, and scapulo-humeral rhythm.
We will try to keep it updated as much as possible. If you have any suggestions, please let us know.

We assume the continuity of the data between articles. For example, if the same data is used in two articles, and some information is missing in the last one, we pick the information in the previous article.

# Still a work in progress but citable
Moissenet, F., Puchaud, P., Naaïm, A., Holzer, N., & Begon, M. (2024). Spartacus-shoulder-kinematics-dataset/shoulder-kinematics congress (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.11455521

```
@software{moissenet_2024_11455521,
  author       = {Moissenet, Florent and
                  Puchaud, Pierre and
                  Naaïm, Alexandre and
                  Holzer, Nicolas and
                  Begon, Mickael},
  title        = {{Spartacus-shoulder-kinematics-dataset/shoulder- 
                   kinematics congress}},
  month        = jun,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.11455521},
  url          = {https://doi.org/10.5281/zenodo.11455521}
}
```

# Exporting the unified dataset
The whole purpose of spartacus is to merge by correcting, aligning, frames and euler sequences of different datasets. 

```python3
from spartacus import Spartacus
  
# Load the dataset using Spartacus
spartacus_dataset = Spartacus.load()

# Export the dataset to the desired format
spartacus_dataset.export('your_path/spartacus.csv')
      
# Return the corrected data values for further analysis
dataframe = spartacus_dataset.corrected_confident_data_values
```
## Data Structure final file The CSV file should have the following columns:

There is one line per measured data point in the dataset, where each line represents a specific measurement 
within a given study or experiment. 
The dataset includes the following columns:

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

# Raw dataset Structure
This repository is organized into three main sections: Data Folder, Dataset of Datasets, and Joint Data. 
This structured organization is crucial for maintaining and normalizing the databases, 
making it easier to manage and analyze the data consistently and avoid to much redundancy.

1. The **Data** folder contains the raw and processed data files. 
Each subfolder within this directory corresponds to a specific study (e.g., `#1_Begon_et_al`). 
. These subfolders are named according to the study authors and contain CSV files with detailed biomechanical data from the experiments.
For example, within the `#3_Chu_et_al` folder, you might find a file named ST_medRotation_elevationFrontal.csv. This file includes two columns:
First column: Represents the thoracohumeral angle.
Second column: Represents an Euler Angle.
The file naming convention is also indicative of the data it contains:
'ST' stands for Scapulothoracic.
'medRotation' stands for Medial Rotation.
'elevationFrontal' stands for frontal plane elevation.
Here’s an example snippet from the `ST_medRotation_elevationFrontal.csv` file:
```
29.774,-13.416
39.527,-17.1201
49.812,-20.7742
59.564,-24.2803
69.671,-27.6376
80.309,-31.1923
90.327,-34.3516
100.519,-36.7187
110.005,-39.8785
119.756,-42.741
129.949,-45.5041
140.056,-49.0099 
```

2. The Dataset of Datasets `dataset_of_datasets.csv` is a meta-dataset that provides a high-level overview 
of all the studies included in the repository. 
This dataset contains summary details like the study authors, publication year, DOI, experimental methods, 
and specific attributes related to shoulder movement data.

3. The Joint Data `dataset_clean_of_joint_data.csv` section includes processed datasets focusing 
on joint movements, particularly shoulder kinematics. Each row within the Joint Data provides 
detailed biomechanical parameters such as joint angles, movement types, and postural data, 
which are linked to the raw data available in the Data Folder for each degree of freedom.

## Dataset colums - `dataset_of_datasets.csv`
The dataset includes the following columns:

- **dataset_id**: A unique identifier for each dataset entry.
- **dataset_authors**: The authors of the study associated with the dataset.
- **dataset_year**: The year the study was published.
- **dataset_doi**: Digital Object Identifier (DOI) links to the associated publications.
- **in_vivo**: Indicates whether the experiment was conducted in vivo (True) or not (False).
- **experimental_mean**: The experimental method used such as intra cortical pins, biplane x-ray fluoroscopy, single-plane x-ray fluoroscopy, 4DCT, etc..
- **number_of_shoulders**: The number of shoulders analyzed in the study.
- **type_of_movement**: The type of movement analyzed (e.g., dynamic, quasi-static).
- **active**: Indicates whether the movement was active (True) or passive (False).
- **posture**: The posture of the subjects during the experiment (e.g., standing, sitting).
- **thorax_is_global**: Specifies if the thorax is considered the global reference frame, common practice with imaging systems.
- **side_as_right**: Indicates if the raw data represents the right side (True) even if they mentioned they captured it on left shoulders.

### Common Anatomical Data Elements:
For each of the anatomical structures — thorax, humerus, scapula, and clavicle —the following elements are defined:

- **_correction_method**: The method used for correcting the specific anatomical structure's data.
- **_origin**: The origin point used for the related measurements of the anatomical structure.
- **_x_direction**, **_y_direction**, **_z_direction**: The directions of the x, y, and z axes, respectively, for the anatomical structure.

#### ISB Humerus Example

| **Column**              | **Origin and Direction** | **Description**                                                                 |
|-------------------------|---------------------|---------------------------------------------------------------------------------|
| **humerus_origin**      | GH                  | The Glenohumeral joint center, used as the origin point for humerus measurements. |
| **humerus_x_direction** | `vec(GH>EL)^vec(GH>EM)` | The cross product between the normalized vector from GH to EL and the normalized vector from GH to EM. |
| **humerus_y_direction** | `vec((EL+EM)/2>GH)` | The normalized vector from the midpoint between EL and EM to GH.                  |
| **humerus_z_direction** | `x^y`               | The cross product between the X and Y directions.                                 |

- vec: normalized vector
- `^`: cross product
- `>`: from to
- GH: Glenohumeral joint center, the central point of rotation for the humerus.
- EL: Lateral Epicondyle, a bony prominence on the outer part of the humerus near the elbow.
- EM: Medial Epicondyle, a bony prominence on the inner part of the humerus near the elbow.
- X^Y: Represents the cross product between vectors X and Y, ensuring the resulting direction is orthogonal to both.

### Computing biomechanical directions from landmarks

A parsing method has been developed to automatically compute the biomechanical direction from landmarks. 
A specific nomenclature and terminology have been chosen where the axis can generally point in the correct direction, 
though not strictly adhering to ISB recommendations. 
However, we can infer the rotation matrix to adjust the orientations of the parent and child segments afterward.

- +posteroanterior: the axis is pointing anteriorly (from posterior to anterior)
- -posteroanterior: the axis is pointing posteriorly
- +mediolateral: the axis is pointing medially (from medial to lateral on the right side)
- -mediolateral: the axis is pointing laterally
- +inferosuperior: the axis is pointing superiorly (from inferior to superior)
- -inferosuperior: the axis is pointing inferiorly

#### Avoiding Terminological Ambiguity
In this nomenclature, we deliberately avoid using terms like "anteroposterior" due to the potential for confusion. 
The term "anteroposterior" could be interpreted differently depending on the context or the perspective of the observer, 
leading to inconsistencies in understanding the direction of the axis. 
By clearly defining each axis with a positive or negative prefix (e.g., +posteroanterior or -posteroanterior), 
we ensure that the direction is explicitly stated, reducing the risk of misinterpretation.


## Joint Data Columns - `dataset_clean_of_joint_data.csv`

The joint data file, `dataset_clean_of_joint_data.csv`, contains detailed and processed biomechanical data focused on shoulder kinematics. Each column represents key biomechanical parameters or metadata related to shoulder movements captured during the experiments. Below is a detailed explanation of the columns included in this dataset:

- **dataset_id**: A unique identifier for each dataset entry.
- **dataset_authors**: The authors of the study associated with the dataset.
- **humeral_motion**: Describes the type of humeral motion, such as frontal or sagittal plane elevation.
- **thoracohumeral_sequence**: The sequence of thoracohumeral rotations, typically expressed in Euler angles (e.g., `yx'y''`).
- **thoracohumeral_angle**: The specific thoracohumeral angle being measured (e.g., `y'`). 
When it is `angle(yt, yh)`, we thought it was an angle between two vectors. When it is `controlled by operator`, it was probably a goniometer.
- **joint**: The joint being analyzed (e.g., glenohumeral, scapulothoracic, acromioclavicular, sternoclavicular).
- **parent**: The parent segment of the joint (e.g., scapula, thorax).
- **child**: The child segment of the joint (e.g., humerus, scapula).
- **euler_sequence**: The sequence of rotations used in Euler angle calculations for the joint (e.g., `yx'y''`).
- **rotation_absolute**: Indicates if the rotation data is presented as absolute values (True/False).
- **origin_displacement**: The displacement of the origin in the coordinate system. Sometimes different.
- **displacement_cs**: The displacement coordinate system. Sometimes different.
- **displacement_absolute**: Indicates if the displacement data is presented as absolute values (True/False).
- **is_data_mean**: Specifies whether the data represents mean values across trials (True/False).
- **shoulder_id**: A unique identifier for each shoulder analyzed in the study.
- **side**: Specifies the side of the body (left or right) that the data represents.
- **source_extraction**: The source from which the data was extracted, such as "authors table" or "engauged".
- **folder**: The folder containing the original data files (e.g., `#1_Begon_et_al`).
- **dof_1st_euler**: The file name or path of the data representing the first degree of freedom in the Euler sequence.
- **dof_2nd_euler**: The file name or path of the data representing the second degree of freedom in the Euler sequence.
- **dof_3rd_euler**: The file name or path of the data representing the third degree of freedom in the Euler sequence.
- **dof_translation_x**: The file name or path of the data representing the x-axis translation component.
- **dof_translation_y**: The file name or path of the data representing the y-axis translation component.
- **dof_translation_z**: The file name or path of the data representing the z-axis translation component.

## Merging Dataset colums and Joint data colums

In this example, we demonstrate how to merge columns using the Spartacus library.
```python3
from spartacus import Spartacus

sp = Spartacus.load(check_and_import=False)
sp.dataframe.to_csv("merged_dataframe.csv")
```
It will duplicate the colums of 2 (Dataset colums) in 3 (Joint data colums).

