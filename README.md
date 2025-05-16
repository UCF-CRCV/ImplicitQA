# ImplicitQA

In this repository, we add the evaluation scripts on ImplicitQA
## Install
```
conda create -n implicit python==3.9 -y
conda activate implicit
pip install torch torchvision torchaudio
mamba install ffmpeg -c conda-forge
ffmpeg -decoders | grep -i nvidia
pip install torchcodec --index-url=https://download.pytorch.org/whl/cu124
pip install vllm
pip install pandas
```


## Evaluation 
`python eval_seq.py --input-dir <path to data> --model-type <model_type> --num-frames 16`
