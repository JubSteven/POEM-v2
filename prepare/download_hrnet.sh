export REPO_DIR=$PWD
if [ ! -d $REPO_DIR/checkpoints ] ; then
    mkdir -p $REPO_DIR/checkpoints
fi
BLOB='https://datarelease.blob.core.windows.net/metro'

wget -nc $BLOB/models/hrnetv2_w40_imagenet_pretrained.pth -O $REPO_DIR/checkpoints/hrnetv2_w40_imagenet_pretrained.pth
wget -nc $BLOB/models/hrnetv2_w64_imagenet_pretrained.pth -O $REPO_DIR/checkpoints/hrnetv2_w64_imagenet_pretrained.pth