# Mic_Hack_2025_Penn

This project focuses on classifying defects and elements in the Quanternary MoSWSe Alloy system data from Jordan Hachtel. It uses extracted atom positions to compute nearest neighbor distances and angles for each atom. Defects are identified and classified using this local bonding geometry. Then, a combination of intensity and lattice site analysis predict the elemental identity of each atom. 

The broad workflow is as follows:\
Step 0: Download the 3D stacks image dataset from (https://doi.ccs.ornl.gov/dataset/78833284-28a4-59da-9790-3624a6f8b9e3)\
Step 1: Run batch_processing_hack_datset.py on the folder containing dm3 files\
Step 2: Run defect_classification.py\
Step 3: Run intensity_classification.py\
Step 4: Run summary_plots.py

## Atom Detection

Atoms positions are extracted using the a blob identification Difference of Gaussians (DoG) method. DoG is a spatial band-pass filter that uses a 2D Gaussian kernel $G_\sigma$ with standard deviation $\sigma$ convolved with a 2D image $I$. A Gaussian blurred image $G_{\sigma_1} \cdot I$ is subtracted from another Gaussian blurried image $G_{\sigma_2} \cdot I$ where $\sigma_2 > \sigma_1$. The ratio $\sigma_2:\sigma_1 = k$ is generally set at a value of 1.6 to approximate the computationally intensive Laplacian of Gaussian (LoG) method. The complete expression reads

$$\text{DoG}(x,y) = G_{\sigma_1} \cdot I - G_{\sigma_2} \cdot I$$
$$\sigma_2:\sigma_1 = k = 1.6$$

A range of acceptable $\sigma$ is provided to constrain blob detection to features approximately the size of an atom. Additionally, a threshold is used on $|\text{DoG}(x,y)|$ to tune it's detection sensitivity. 

Once a list of atomic positions has been provided, K nearest neighbors (KNN) where $K=3$ are identified using a KDTree to optimize performance. The distances $l_1, l_2, l_3$ from the center atom and its nearest neighbors are calculated as well as angles the nearest neighbor and the second and third nearest neighbors. These coordinates, nearest neighbor lengths, angles and summary statistics are saved as a csv for the image.

Below is an example of detected atom positions on "3D Stack align2"

<img width="500" height="500" alt="02_atoms3D Stack align2" src="https://github.com/user-attachments/assets/ea4194c3-c79a-4340-9422-cefe858d60de" />

## Defect Classification

We classify lattice defects at the atom level using local bonding geometry around each detected atomic site. Vacancy-related classes label atoms adjacent to vacancies rather than the vacancy sites themselves. This is intentional because (i) vacancy sites are not directly observed in the atom-position list and (ii) assigning defect labels to atoms allows a direct one-to-one comparison with per-atom element labels in the later stage of the pipeline.

### Defect types

In a hexagonal (honeycomb-like) lattice, the ideal local environment for each atom has three nearest-neighbor (NN) bond lengths close to the ideal bond length \(a\). Deviations from this pattern are used to classify defects into:

- No Defect
- Interstitial
- One Adjacent Vacancy
- Two Adjacent Vacancies
- Three Adjacent Vacancies
- Others (Distortion)

These defect motifs and their local geometric signatures are motivated by prior first-principles and ab initio studies of vacancy and defect configurations in monolayer MoS<sub>2</sub>. [1,2]

<img height="500" alt="Defect types" src="https://github.com/vnpawan/Mic_Hack_2025_Penn/blob/158c9791ea9b27b6b0831d8878758b124a920baa/readme_figure/my_structure.jpg" />

### Classification workflow

For each atom, we compute the three NN distances $(l_1, l_2, l_3)$ and sort them from smallest to largest. We use:

- $a = 183 \mathrm{pm}$ (ideal bond length)
- $\varepsilon = 0.25a$
- $\delta = 0.4a$
- $\Delta = 0.4a$

Classes are assigned sequentially (priority order) so that each atom receives exactly one label:

Class 1: No Defect  
$\max\left(|l_1-a|,|l_2-a|,|l_3-a|\right) < \varepsilon$

Class 2: Interstitial  
A too-short NN distance indicates local crowding consistent with an interstitial-like environment.  
$\min(l_1,l_2,l_3) < a - \delta$

Class 3: One Adjacent Vacancy  
Two NN bonds remain close to $a$, while the third is significantly longer.  
$\max\left(|l_1-a|,|l_2-a|\right) < \varepsilon \\land\ l_3 > a + \Delta$

Class 4: Two Adjacent Vacancies  
Only the closest NN bond remains near $a$, while the other two are significantly longer.  
$|l_1-a| < \varepsilon \\land\ \min(l_2,l_3) > a + \Delta$

Class 5: Three Adjacent Vacancies  
All three NN distances are significantly longer than $a$.  
$\min(l_1,l_2,l_3) > a + \Delta$

Class 6: Others (Distortion)  
Any atom not captured by Classes 1–5 is labeled as Distorted Structure (Others), representing local environments that deviate from the idealized vacancy/interstitial motifs (for example, lattice distortion, complex defects, boundaries, or imperfect neighborhoods due to detection noise).

<img height="500" alt="Classification workflow" src="https://github.com/vnpawan/Mic_Hack_2025_Penn/blob/158c9791ea9b27b6b0831d8878758b124a920baa/readme_figure/my_logic_flow.jpg" />

### Practical notes

- Per-atom classification: vacancy-related classes label atoms neighboring a vacancy rather than the vacancy sites themselves, which also enables direct comparison against per-atom element labels later.
- Edge filtering: atoms within a configurable margin of the image boundary are excluded to avoid incomplete NN neighborhoods near edges.
- Outputs (per dataset): the pipeline saves (i) labeled overlay images on the enhanced DM image, (ii) per-class overlay images, (iii) a bar chart of class counts (title includes total atoms), (iv) a histogram of `Mean_Distance_pm` separated by class, and (v) a compact CSV with (`Atom_Index`, `Y_Position_px`, `X_Position_px`, `Defect_Class`).

### Example result

Example overlay showing atom-level defect labels for dataset `3D Stack align2`:

<img height="500" alt="Defect classification result on 3D Stack align2" src="https://github.com/vnpawan/Mic_Hack_2025_Penn/blob/158c9791ea9b27b6b0831d8878758b124a920baa/3D%20Stack%20align2_defect_class/3D%20Stack%20align2_no_defect_interstitial_overlay.png" />

## Intensity-based Element Classification

Segregation of the lattice sites into A and B sites is needed to narrow down the intensity-based element identification of the metals (M: W or Mo) and chalcogens (X: S or Se) in our Quaternary Transition Metal Dichalogenide (TMD) system. Each lattice point was classified into A-Site or B-Site depending on Coordination of Nearest Neighbors:

- A-Site corresponds to 2 lattice points above, and 1 below.
- B-Site corresponds to 2 lattice points below, and 1 above.

Note: This part of the code was used by the author(s) of the previous work on this dataset.

<img width="468" height="236" alt="image" src="https://github.com/user-attachments/assets/c061980f-1234-468b-9a90-1869985a8b57" />



Local patch- based normalization was done on the images to reduce background intensity arising from wrinkling of the monolayer TMD sample.

<img width="468" height="228" alt="image" src="https://github.com/user-attachments/assets/6991a81f-91fb-4ae4-9a9c-c6f36c5d1c26" />

### A-site intensity histogram fitted with 2 Gaussian peaks (representing 2 element types)

First peak corresponds to Mo (Z = 42), second is W (Z = 74). Individual intensities are proposonal to Z^2 based on the working principle of HAADF-STEM.

<img width="468" height="390" alt="image" src="https://github.com/user-attachments/assets/30a97ae7-09cf-4bdd-985a-1e0bee97956e" />

### B-site intensity histogram fitted with 3 Gaussian peaks (representing 2 element types and their combination)

First peak corresponds to S2 (Z = 16), second one is S + Se, and third one is Se2 (Z = 34).

<img width="468" height="390" alt="image" src="https://github.com/user-attachments/assets/3a9c471c-5b77-4fcd-9b4e-dc69e16c1803" />

Each lattice point with its intensity value was correlated with the elements based on z-scores from each deconvoluted peak.

Z-score was calculated as (intensity of lattice point - mean of a particular Gaussian peak)/standard deviation of that particular Gaussian peak.


### Overall workflow

Step 1: Data Aggregation

- Reads atom position data from multiple CSV files (generated by a previous batch processing step)
- Loads corresponding images and extracts intensity values for detected atoms
- Aggregates intensity data from selected files to create a global dataset

Step 2: Global Histogram Fitting

- Separates atoms into two sublattices (A-site and B-site) based on nearest-neighbor distances
- Fits A-site intensity histogram with 2 Gaussian peaks (representing 2 element types)
- Fits B-site intensity histogram with 3 Gaussian peaks (representing 2 element types and their combination)
- Creates global fitting parameters to be applied consistently across all files

Step 3: Individual File Processing

For each image file:

- Loads the image and applies local normalization
- Reads atom positions from CSV
- Sorts atoms into A-site and B-site sublattices
- Classifies each atom into one of 5 element types based on intensity using global fit parameters
- Updates the CSV file with element classifications and file IDs
- Creates visualization overlays showing element assignments

Step 4: Results Aggregation

- Combines all classified atoms into a master CSV file
- Tracks which file each atom came from using file IDs
- Generates summary statistics including element distributions

The final figure shows the distribution of different elements across the various defects.

<img width="3568" height="2068" alt="element_by_defect_stacked_bar" src="https://github.com/user-attachments/assets/fd4152f5-3cf7-484d-b039-d7e2d6695bfc" />


## References

[1] Komsa, H. P., & Krasheninnikov, A. V. (2015). Native defects in bulk and monolayer MoS 2 from first principles. Physical Review B, 91(12), 125304.

[2] Santosh, K. C., Longo, R. C., Addou, R., Wallace, R. M., & Cho, K. (2014). Impact of intrinsic atomic defects on the electronic structure of MoS2 monolayers. Nanotechnology, 25(37), 375703.







