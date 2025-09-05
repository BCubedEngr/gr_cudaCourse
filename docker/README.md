# To build Docker image locally
You must set up your Artifactory secrets as described [here](https://gitlab.bcubed.local/artifactory/setup#secrets-file)

Ensure you run `docker login` prior to building.

```bash
./docker-build.sh
```

# To launch a new Development Container

Ensure you run `xhost +local:docker` prior to running.

```bash
./docker-run.sh ${CONTAINER_NAME}
```
