# AWB-Lib

[![Smoke Test](https://github.com/JoeyTeng/AWB-Lib/actions/workflows/smoke_test.yml/badge.svg)](https://github.com/JoeyTeng/AWB-Lib/actions/workflows/smoke_test.yml)

Implementation of a bunch of Automatic White-Balancing (AWBE algorithms), using JAX. Comparisons and experiments can be found in `docs` folder.

[Implemented Algorithms](#implemented-algorithms)
| [Example Results](#example-results)
| [Notes](#notes)
| [License](#license)

## Implemented Algorithms

1. Gray World (GW), based on gray world assumption. See `awblib.gw`.
2. Colour Histogram Stretching (CHS). See `awblib.chs`.
3. Average Equalization and Threshold (AAET). See `awblib.aaet`.
4. Histogram Matching (AWB-HM), based on histogram matching. See `awblib.hm`.
5. Dynamic Histogram Matching (AWB-DHM), based on histogram matching and channel selection. See `awblib.dhm`.

### AWB-DHM

> T. Gollanapalli, V. R. Peddigari and P. S. Madineni, "Auto white balance using dynamic histogram matching for AMOLED panels," 2017 IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia), Bengaluru, India, 2017, pp. 41-46, doi: 10.1109/ICCE-ASIA.2017.8307848.

Explanations about our implementation is [here](docs/dhm/README.md).

### AWB-HM

> Chengqiang Huang, Qi Zhang, Hui Wang, and Songlin Feng, "A Low Power and Low Complexity Automatic White Balance Algorithm for AMOLED Driving Using Histogram Matching," J. Display Technol. 11, 53-59 (2015)

Explanations about our implementation is [here](docs/hm/README.md). This implementation largely reproduced the results.

### CHS (Colour Histogram Stretching)

> S. Wang, Y. Zhang, P. Deng and F. Zhou, "Fast automatic white balancing method by color histogram stretching," 2011 4th International Congress on Image and Signal Processing, Shanghai, China, 2011, pp. 979-983, doi: 10.1109/CISP.2011.6100338.

The implementation largely reproduced the results.

### AWBAAET (Average Equalizatin and Threshold)

> Shen-Chuan Tai, Tzu-Wen Liao, Yi-Ying Chang and Chih - Pei Yeh, "Automatic White Balance algorithm through the average equalization and threshold," 2012 8th International Conference on Information Science and Digital Content Technology (ICIDT2012), Jeju, Korea (South), 2012, pp. 571-576.

## Example Results

See [`docs`](docs/README.md) for more.

![Report](docs/assets/report.png)

## Notes

Actually no paper has mentioned that if they are invariant to colour space or they depend on a  specific colour space. As far as I know, most non-ML image processing algorithms require linear RGB; this processing is not done in this module yet.

The difference is performance (qualitatively, measured by overlapping area (OA) in colour histogram) is largely due to the discrete nature of the small image and thus the resultant histogram is often very discrete.

## License

[Apache-2.0](LICENSE)
