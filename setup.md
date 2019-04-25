

## Initial Setup
The first thing to do in an aws instance is to clone the repo:  
```bash
$ git clone https://github.com/mosqueteiro/scalable_DS_envs.git && \
cd scalable_DS_envs
```

Switch to the `aws-gpu` branch then run the aws_setup.sh script:
```bash
$ git checkout aws-gpu
$ bash aws_setup.sh
```

## PostgreSQL setup  
We need to load our SQL database with all the information from the annotations.  



## EBS storage volume setup  
_from_ [_AWS docs - ebs volumes_](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html)  
1. List the block devices connected to instance and determine which _block name_ is the EBS storage (it should be the unmounted block)  
```bash
ubuntu@ip-xxx-xx-xx-xx:~$ lsblk
NAME        MAJ:MIN RM SIZE RO TYPE MOUNTPOINT
loop0         7:0    0  91M  1 loop /snap/core/6350
loop1         7:1    0  18M  1 loop /snap/amazon-ssm-agent/930
nvme0n1     259:0    0  50G  0 disk
nvme1n1     259:1    0   8G  0 disk
└─nvme1n1p1 259:2    0   8G  0 part /
```

2. New EBS volumes do not have a filesystem  
```bash
ubuntu@ip-xxx-xx-xx-xx:~$ sudo file -s /dev/nvme0n1
/dev/nvme0n1: data
```  
so we need to make one  
```bash
ubuntu@ip-xxx-xx-xx-xx:~$ sudo mkfs -t xfs /dev/nvme0n1
```  
now the block should have a file system  
```bash
ubuntu@ip-xxx-xx-xx-xx:~$ sudo file -s /dev//nvme0n1
/dev//nvme0n1: SGI XFS filesystem data (blksz 4096, inosz 512, v2 dirs)
```

3. Now we need to mount the EBS filesystem to a directory in our current filesystem  
```bash
ubuntu@ip-xxx-xx-xx-xx:~$ mkdir data
ubuntu@ip-xxx-xx-xx-xx:~$ sudo mount /dev/nvme0n1 data/
```
4. Now download the data to the data directory, unzip it and then remove the zip file. First we give ownership of this mount dir to our user and install unzip (base AWS AMIs don't come with unzip for some reason).  
```bash
ubuntu@ip-xxx-xx-xx-xx:~$ sudo chown -R $USER:$USER data/
ubuntu@ip-xxx-xx-xx-xx:~$ sudo apt install unzip
ubuntu@ip-xxx-xx-xx-xx:~$ cd data/
ubuntu@ip-xxx-xx-xx-xx:~/data$ wget http://images.cocodataset.org/zips/train2017.zip
ubuntu@ip-xxx-xx-xx-xx:~/data$ unzip train2017.zip
ubuntu@ip-xxx-xx-xx-xx:~/data$ rm train2017.zip
```

5. Our data is now saved to an EBS volume
