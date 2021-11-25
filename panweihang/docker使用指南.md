通过mmdetection3D镜像建立docker

```shell
docker run -itd --name panweihang mmdetection3D
```

`mmdetection3D`是image的名字

进入一个docker容器

```shell
docker exec -it panweihang /bin/bash
```

如果docker容器没有在运行，需要重新start

```shell
docker start panweihang
```

重启容器

```shell
docker restart panweihang
```

