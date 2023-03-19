#  stable-diffusion-webui-docker

Docker with `torch nightly`, `xformers` and a lot of other dependendies.

Building docker:
```
DOCKER_BUILDKIT=1 sudo docker build  -t styler00dollar/stable_diffusion:latest . 
```

Running docker:
```
sudo docker pull styler00dollar/stable_diffusion:latest
sudo docker run --privileged --gpus all -it --rm -v /home/user/stable-diffusion-webui:/workspace/ --network host styler00dollar/stable_diffusion:latest
python launch.py --xformers --force-enable-xformers --precision autocast --xformers-flash-attention
```
