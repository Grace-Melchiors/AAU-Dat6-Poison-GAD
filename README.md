# AAU-Dat6-Poison-GAD
## Description
This project is made alongside a bachelor project report for AAU, focusing increasing the robustness of Dominant against a greedy camouflaging attack.
It takes a dataset and injects it with anomalies, it then detects the performance of GAD methods at different iterations of the camouflaging attack.
It provides tools to visualize this performance, and can be adjusted as needed.

## Table of Contents
- [Important Files](#important-files)
- [Contributing](#contributing)
- [License](#license)

## Important Files
The files central to the project include:
- [GMS Poison](gad_adversarial_robustness/poison/greedy.py)
- [Base Dominant](gad_adversarial_robustness/gad/dominant/dominant_cuda_v2.py)
- [Jaccard Dominant](gad_adversarial_robustness/gad/dominant/dominant_cuda_Jaccard_similarity.py)
- [Soft Medoid Dominant](gad_adversarial_robustness/gad/dominant/dominant_cuda_medoid.py)
- [CamoBlock Dominant](gad_adversarial_robustness/gad/dominant/dominant_cuda_camoblock.py)

## Contributing
This project is developed by
- Andreas Worm Holt
- Grace Melchiors
- Karen Andersen

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

