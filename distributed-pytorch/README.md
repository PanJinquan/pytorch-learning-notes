# distributed-pytorch

modified version of src

python main.py /home/zhangzhaoyu --dist-rank 0

python main.py /home/zhangzhaoyu --dist-rank 1

# 
> https://github.com/pytorch/examples/tree/master/imagenet
> https://github.com/zzy123abc/distributed-pytorch
> https://blog.csdn.net/qq_20791919/article/details/79057871


> os.environ['NCCL_SOCKET_IFNAME'] = 'bond0'
> os.environ['MASTER_ADDR'] = 'localhost'
> os.environ['MASTER_PORT'] = '1234'
## service 
> python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 --node_rank=0 distributed-pytorch.py
Namespace(arch='resnet18', batch_size=8, ckpt_save='./models',
 data='../dataset/images', dist_backend='nccl', dist_rank=2, 
 dist_url='env://', distributed=True, epochs=90, evaluate=False, 
 local_rank=5, lr=0.1, momentum=0.9, pretrained=False, 
 print_freq=10, resume='', start_epoch=0, 
 weight_decay=0.0001, workers=4, world_size=8)

## loacal 
>  distributed-pytorch.py 
