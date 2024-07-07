# FairViT: Fair Vision Transformer via Adaptive Masking
The code of the paper "FairViT: Fair Vision Transformer via Adaptive Masking", this paper will appear at ECCV 2024.

Step 1: Create the conda environment through vit2024.yaml. The command is "conda env create -f vit2024.yaml".

Step 2: Download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place the decompressed Anno, Eval and img_align_celeba in "../nfs3/datasets/celeba". The preferred file structure is:

- nfs3 - datasets - celeba - Anno  
|                           - Eval  
|                           - img_align_data  
- FairViT - README.md  
          - ...  

Step 3: You can run the code by "python finetune_transformer.py" for debugging. To implement the default tasks in the paper, here are some quick commands:

python finetune_transformer.py --feature Smiling_Male -e 30 -r 1 -ta Smiling -sa Male -d celeba --gpu_en 0 -g 0.5 -s 19 -alr 3 

python finetune_transformer.py --feature Attractive_Male -e 30 -r 1 -ta Attractive -sa Male -d celeba --gpu_en 0 -g 0.5 -s 19 -alr 3 

python finetune_transformer.py --feature Attractive_Brown_Hair -e 30 -r 1 -ta Attractive -sa Brown_Hair -d celeba --gpu_en 0 -g 0.5 -s 19 -alr 3

To get further helps, please refer to "help" options in finetune_transformer.py.