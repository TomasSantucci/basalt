# Basalt for Monado

This is a fork of [Basalt](https://gitlab.com/VladyslavUsenko/basalt) improved
for tracking XR devices with
[Monado](https://gitlab.freedesktop.org/monado/monado). Many thanks to the
Basalt authors.

## Installation

- **Prebuilt (Ubuntu/Raspberry/Radxa)**: Download [latest .deb](https://gitlab.freedesktop.org/mateosss/basalt/-/releases) and install with

  ```bash
  sudo apt install -y ./basalt-monado-*.deb
  ```

- **From source (Linux)**

  ```bash
  git clone --recursive https://gitlab.freedesktop.org/mateosss/basalt.git
  cd basalt && ./scripts/install_deps.sh
  cmake --preset library # use "development" instead of "library" if you want extra binaries and debug symbols
  sudo cmake --build build --target install
  ```

- **From source (Windows)**: See the [build guide](doc/monado/Windows.md) for Windows.

## Usage

If you want to run OpenXR application with Monado, you need to set the
environment variable `VIT_SYSTEM_LIBRARY_PATH` to the path of the basalt library.

By default, Monado will try to load the library from `/usr/lib/libbasalt.so` if
the environment variable is not set.

If you want to test whether everything is working you can download a short dataset with [EuRoC (ASL) format](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) format like [`MOO09_short_1_updown`](https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MO_odyssey_plus/MOO_others/MOO09_short_1_updown.zip?download=true) from the [Monado SLAM datasets](https://huggingface.co/datasets/collabora/monado-slam-datasets):

```bash
wget https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MO_odyssey_plus/MOO_others/MOO09_short_1_updown.zip
unzip MOO09_short_1_updown.zip
```

- **Try it standalone with a dataset (requires extra binaries)**

  ```bash
  basalt_vio --show-gui 1 --dataset-path MOO09_short_1_updown/ --dataset-type euroc --cam-calib /usr/share/basalt/msdmo_calib.json --config-path /usr/share/basalt/msdmo_config.json
  ```

- **Use a RealSense camera without Monado (requires extra binaries)**
  You'll need to calibrate your camera if you want the best results but meanwhile you can try with these calibration files instead.

  - RealSense D455 (and maybe also D435)

    ```bash
    basalt_rs_t265_vio --is-d455 --cam-calib /usr/share/basalt/d455_calib.json --config-path /usr/share/basalt/default_config.json
    ```

  - Realsense T265: Get t265_calib.json from [this issue](https://gitlab.com/VladyslavUsenko/basalt/-/issues/52) and run

    ```bash
    basalt_rs_t265_vio --cam-calib t265_calib.json --config-path /usr/share/basalt/default_config.json
    ```

- **Try it through `monado-cli` with a dataset**

  ```bash
  monado-cli slambatch MOO09_short_1_updown/ /usr/share/basalt/msdmo.toml results
  ```

- **Try it with `monado`, a dataset, and an OpenXR app**

  ```bash
  # Run monado-service with a fake "euroc device" driver
  export EUROC_PATH=MOO09_short_1_updown/ # dataset path
  export EUROC_HMD=false # false for controller tracking
  export EUROC_PLAY_FROM_START=true # produce samples right away
  export SLAM_CONFIG=/usr/share/basalt/msdmo.toml # includes calibration
  export SLAM_SUBMIT_FROM_START=true # consume samples right away
  export XRT_DEBUG_GUI=1 # enable monado debug ui
  monado-service &

  # Get and run a sample OpenXR application
  wget https://gitlab.freedesktop.org/wallbraker/apps/-/raw/main/VirtualGround-x86_64.AppImage
  chmod +x VirtualGround-x86_64.AppImage
  ./VirtualGround-x86_64.AppImage normal
  ```

- **Use a real device in Monado**.

  When using a real device driver you might want to enable the `XRT_DEBUG_GUI=1` and `SLAM_UI=1` environment variables to show debug GUIs of Monado and Basalt respectively.

  Monado has a couple of drivers supporting SLAM tracking (and thus Basalt). Most of them should work without any user input.

  - WMR ([troubleshoot](doc/monado/WMR.md))
  - Rift S (might need to press "Submit to SLAM", like the Vive Driver).
  - Northstar / DepthAI ([This hand-tracking guide](https://monado.freedesktop.org/handtracking) has a depthai section).
  - Vive Driver (Valve Index) ([read before using](doc/monado/Vive.md))
  - RealSense Driver ([setup](doc/monado/Realsense.md)).

## Development

If you want to set up your build environment for developing and iterating on Basalt, see the [development guide](doc/Development.md).

I am inclining into merging the timing and eval pipelines

- On one hand I would want the eval pipeline to be parallelized
- On the other I dont know of a way to do this in gitlab while at the same time keeping the timing stuff isolated
- I can get memory timing info and everything from everything
- Maybe I could just have specified evalsets for runs that are meant to happen in a separte stage isolated but even then I dont know if I can isolate the runner

- Do stacked timing plot with each stage one after the other
  - Support multiple timing files by putting each bar of each system next to each other
  - Add uncertainty lines

- fix chrome encoding
- investigate the difference in trajectory output between main branch and this timing branch (I edited addPoints)
- Add --num-threads option to CI





-------------






```cpp
  void addPoints() {
    ManagedImagePyr<uint16_t>& pyr0 = pyramid->at(0);
    Keypoints kpts0 = addPointsForCamera(0);
    Masks& ms0 = transforms->input_images->masks.at(0);

    if (config.optical_flow_recall_enable) kpts0.insert(recalls[0].begin(), recalls[0].end());

    // Match features on areas that overlap with cam0 using optical flow
    for (size_t i = 1; i < getNumCams(); i++) {
      Masks& ms = transforms->input_images->masks.at(i);
      Keypoints& mgs = transforms->matching_guesses.at(i);
      ManagedImagePyr<uint16_t>& pyri = pyramid->at(i);
      Keypoints kpts;
      SE3 T_c0_ci = calib.T_i_c[0].inverse() * calib.T_i_c[i];
      trackPoints(pyr0, pyri, kpts0, kpts, mgs, ms0, ms, T_c0_ci, 0, i);
      addKeypoints(i, kpts);
    }

    if (!config.optical_flow_detection_nonoverlap) return;

    // Update masks and detect features on area not overlapping with cam0
    for (size_t i = 1; i < getNumCams(); i++) {
      Masks& ms = transforms->input_images->masks.at(i);
      ms += cam0OverlapCellsMasksForCam(i);
      Keypoints kpts_no = addPointsForCamera(i);
    }
  }
```


# TODO@mateosss Add perf flamegraph

sudo perf record -F 99 -g ./build/basalt_vio --dataset-path $msdmo/MOO_others/MOO02_hand_puncher_2 --cam-calib data/msd/msdmo_calib.json --config-path data/msd/msdmo_config.json --show-gui 0 --deterministic 1
sudo perf script > out.perf
./FlameGraph/stackcollapse-perf.pl out.perf > out.folded
./FlameGraph/flamegraph.pl out.folded > flamegraph.svg
