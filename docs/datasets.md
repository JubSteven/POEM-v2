## :floppy_disk: Datasets

The following datasets are used for training. 

- [DexYCB](https://dex-ycb.github.io/)
- [HO3D](https://github.com/shreyashampali/ho3d)
- [Arctic](https://arctic.is.tue.mpg.de/)
- [InterHand](https://mks0601.github.io/InterHand2.6M/)
- [Oakink](https://oakink.net/)
- [FreiHand](https://github.com/lmb-freiburg/freihand)

We have used [webdataset](https://github.com/webdataset/webdataset) to pack the datasets into shards for efficient training. In training POEM, we have also filtered samples with left hand only. 

The packed dataset tars used for training and evaluation will be released soon.

&nbsp;

## :luggage: Assets

Download `mano_v1_2.zip` from the [MANO website](https://mano.is.tue.mpg.de) (Sign in -> Download -> Models & Code), unzip, and copy it to `assets/mano_v1_2`:

```Shell
$ cd assets
$ cp -r path/to/mano_v1_2 assets/
```