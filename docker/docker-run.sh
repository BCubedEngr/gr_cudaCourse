if [ "$#" -eq 0 ]; then
    container_name="gr_dev"
else
    container_name="$1"
fi
docker run -dit --gpus all --runtime nvidia -v ~/git-repos:/git -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority:rw -e "DISPLAY=${DISPLAY:-:0.0}" --name $container_name nvidia/cuda:12.6.3-base-ubuntu22.04-gr-dev bash
docker exec -it $container_name bash
