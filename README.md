# AIrchitect-v2

## Install 

```
conda create -n <name> python=3.8 -y
conda activate <name>
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

We provide the DSE dataset for hardware resource assignment on MAESTRO[1]-modeled accelerator.
The dataset of size 100k is located in ```./dse_dataset``` and no download is required. There are two dataset files: gemm_dataset_*.csv provides DSE input in terms of GEMM workload whereas conv_dataset_*.csv provides DSE input in terms of convolution workloads.

- Columns [```K, C, X, Y, R, S, df```] or [```M, N, GEMM_K, df```]: DSE input, consisting of workload convolution dimensions and dataflow style
    ```
    K: random.randrange(2, 2*256+1, 2)
    C: random.randrange(2, 2*256+1, 2)
    X: random.choice([1] + [i for i in range(8, 8*32+1, 8)])
    Y: random.choice([1] + [i for i in range(8, 8*32+1, 8)])
    R: [1,3,5,7,9]
    S: R
    df: dla, eye, shi
    ```


- Columns [```ConfigID, rewards```] : DSE output, obtained through running ConfuxiuX[2]
    ```
    Config ID: optimal hardware configuration(#PE, Buffer Size) for given DSE input. 
        - #PE:[1,64], Buffer Size:[1,12]
        - ConfigID = #PE * max(Buffer Size) + Buffer Size
    rewards: resulting performance values of the DSE, as well as the optimization goal for ConfuciuX. set to latency in this work.
    
    ```


** To add support for your own DSE data, modify ```dataset.py``` accordingly following a similar dataloader structure for the existing dataset.

For simplicity in this implementation, we assume configID as the label to predict based on a single UOV head, but configID can be decomposed as #PEs and Buffer Size independently following the above formula and trained with two UOV heads as explained in the paper.

Furthermore, rewards can also be used as an additional parameter to improve DSE learning by the model.

## Run Instruction 
1. Stage 1 Encoder Training 

```
python3 main.py  --data ./dse_dataset/conv_dataset_100k.csv --model Transformer --enable_surrogate --alpha 0.2 --save
```
2. Stage 2 Decoder + UOV Training:

```
python3 main_linear.py  --data ./dse_dataset/conv_dataset_100k.csv --model Transformer --enable_surrogate  --classifier Transformer --load_chkpt <path-to-stage1-model> [--ordinal] [--interval]
```

```--ordinal``` and ```--interval``` are optional arguments to enable Unified Regressiona and Classification


## Unified Representation
Our representation of unified ordinal vectors can be found in losses.py

## Pretrained models 
Append with ```--load_chkpt``` to above instruction
```
pretrained_models/stage1_encoder.pth
pretrained_models/stage2_decoder.pth
```

## Citation
If you find this repo useful and use our dataset/artifacts, please cite

```
@article{ramachandran2025uov,
  title={AIrchitect v2: Learning the Hardware Accelerator Design Space through Unified Representations},
  author={Ramachandran, Akshat and Seo, Jamin and Chuang, Yu-Chuan and Itagi, Anirudh and Krishna, Tushar},
  booktitle={2025 Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
  pages={1--7},
  year={2025},
  organization={IEEE}
}
```


## References
[1] H. Kwon, P. Chatarasi, V. Sarkar, T. Krishna, M. Pellauer, and
A. Parashar, “MAESTRO: A data-centric approach to understand reuse,
performance, and hardware cost of DNN mappings,” IEEE Micro, vol. 40,
no. 3, pp. 20–29, 2020.

[2 ]S.-C. Kao, G. Jeong, and T. Krishna, “Confuciux: Autonomous hardware
resource assignment for dnn accelerators using reinforcement learning,”
in 2020 53rd Annual IEEE/ACM International Symposium on Microar-
chitecture (MICRO), 2020, pp. 622–636
