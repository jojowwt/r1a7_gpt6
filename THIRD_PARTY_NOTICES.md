# Third-Party Notices

This project contains or is derived from third-party software and robot assets. Retain the upstream license and notices when redistributing the corresponding files.

| Component | Source | License / note |
|---|---|---|
| RoboICL policy and selected configs | [Mosi-AI/RoboICL](https://github.com/Mosi-AI/RoboICL) | MIT; see `third_party/RoboICL/LICENSE` |
| XPolicyLab image utilities | [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | Upstream license copied to `third_party/RoboICL/third_party/XPolicyLab/LICENSE` |
| RoboDojo IK continuity guard | [Mosi-AI/RoboDojo](https://github.com/Mosi-AI/RoboDojo) | MIT; see `third_party/RoboICL/third_party/RoboDojo/LICENSE` |
| R1-A7 URDF and meshes | Unitree robot description distributed with the R1-A7 integration source | Apache-2.0; see `assets/r1_a7/LICENSE` |
| Unitree SDK2 Python | [unitreerobotics](https://github.com/unitreerobotics) | Installed separately; governed by its upstream license |

RoboICL's selected task JSON files include provenance strings that name the source repository's internal dataset locations. These strings document upstream experiment provenance only; those datasets are not included in this project.

The `robot_teleop` inference hardware interface (R1-A7 DDS control, IK, live camera input, and runtime monitoring) and the setup/documentation authored for this standalone repository are released under the root MIT license where they are original works. VR teleoperation and demonstration-capture code are not included. Files copied or adapted from upstream repositories remain subject to their original terms. This notice is an inventory, not a substitute for upstream license texts; relevant texts are included beside redistributed code.
