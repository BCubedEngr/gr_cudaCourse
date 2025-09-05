# gr_cuda_course

This is designed to help you build your first CUDA accelerated GNU Radio blocks.  This contains a Dockerfile that produces a container that can be used
for building all of the code.  If you wish to work on bare metal, then see the Dockerfile for a list of dependencies to install.

To run the docker image:
```
docker run -it --gpus --runtime nvidia <image_name> bash
```

If you would like X11 forwarding so that you can use GRC run:
```
docker run -it --gpus --runtime nvidia -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY <image_name> bash
```

You may need to run `xhost +` from a terminal outside of the docker container for displays to work.


This course content was developed by [BCubed Engineering](https://bcubed.com).  We are advanced GNU Radio developers and are available for consulting and
trainings.
