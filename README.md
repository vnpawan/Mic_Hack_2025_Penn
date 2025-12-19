# Mic_Hack_2025_Penn

This project focuses on classifying defects and elements in the Quanternary MoSWSe Alloy system data from Jordan Hachtel. It uses extracted atom positions to compute nearest neighbor distances and angles for each atom. Defects are identified and classified using this local bonding geometry. Then, a combination of intensity and lattice site analysis predict the elemental identity of each atom. 

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

## References

[1] Komsa, H. P., & Krasheninnikov, A. V. (2015). Native defects in bulk and monolayer MoS 2 from first principles. Physical Review B, 91(12), 125304.

[2] Santosh, K. C., Longo, R. C., Addou, R., Wallace, R. M., & Cho, K. (2014). Impact of intrinsic atomic defects on the electronic structure of MoS2 monolayers. Nanotechnology, 25(37), 375703.


