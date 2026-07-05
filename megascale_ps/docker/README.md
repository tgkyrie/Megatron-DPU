# Prebuilt Images

Belows are prebuilt docker images, and their associated commands to build. These prebuilt images might not be up-to-date.
You may need to manually build them to get the latest functionalities of MegaScalePS using the dockerfile.

| Docker image | How to build |
| --- | --- |
| megascale_psimage/tensorflow       | docker build -t megascale_psimage/tensorflow . -f Dockerfile --build-arg FRAMEWORK=tensorflow |
| megascale_psimage/pytorch          | docker build -t megascale_psimage/pytorch . -f Dockerfile --build-arg FRAMEWORK=pytorch |
| megascale_psimage/mxnet            | docker build -t megascale_psimage/mxnet . -f Dockerfile --build-arg FRAMEWORK=mxnet |
