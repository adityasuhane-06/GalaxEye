New-Item -ItemType Directory -Force -Path data\raw | Out-Null

curl.exe -L --fail --retry 5 -C - -o data\raw\train.zip "https://huggingface.co/datasets/doron333/change-detection-dataset/resolve/main/train.zip"
curl.exe -L --fail --retry 5 -C - -o data\raw\val.zip "https://huggingface.co/datasets/doron333/change-detection-dataset/resolve/main/val.zip"
curl.exe -L --fail --retry 5 -C - -o data\raw\test.zip "https://huggingface.co/datasets/doron333/change-detection-dataset/resolve/main/test.zip"

Expand-Archive -Force -Path data\raw\train.zip -DestinationPath data\raw\train
Expand-Archive -Force -Path data\raw\val.zip -DestinationPath data\raw\val
Expand-Archive -Force -Path data\raw\test.zip -DestinationPath data\raw\test
