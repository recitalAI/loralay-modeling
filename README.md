# LoRaLay: A Multilingual and Multimodal Dataset for *Lo*ng *Ra*nge and *Lay*out-Aware Summarization

## Environment Setup
~~~shell
conda create -n loralay python=3.8
conda activate loralay 
git clone https://anonymous.4open.science/r/loralay-modeling-1870/
cd loralay-modeling 
pip install -r requirements.txt
pip install -e .
~~~ 

## Fine-tuning

To fine-tune one of the baselines on LoRaLay datasets, run the following:

~~~shell
python experiments/run_summarization.py \
    --output_dir path/to/output/dir \
    --do_train \
    --do_eval \
    --prediction_loss_only \
    --model_type <pegasus|layout-pegasus|bigbird-pegasus|layout-bigbird-pegasus|mbart|layout-mbart|bigbird-mbart|layout-bigbird-mbart> \
    --model_name_or_path pretrained/model/name/or/path \
    --config_name config/name/or/path \
    --tokenizer_name tokenizer/name/or/path \
    --use_fast_tokenizer \
    --dataset_name <arxiv_lay|pubmed_lay|hal|scielo_es|scielo_pt|koreascience> \
    --data_dir path/to/data/dir \
    --max_source_length <1024|3072|4096> \
    --max_target_length 256 \
    --pad_to_max_length \
    --max_steps <50000|74000|100000> \
    --warmup_steps <5000|10000> \
    --learning_rate 1e-4 \
    --adafactor \
    --lr_scheduler_type polynomial \
    --power 0.5 \
    --label_smoothing_factor 0.1 \
    --gradient_checkpointing 
~~~

## Evaluation

To evaluate your model, run the following:

~~~shell
python experiments/run_summarization.py \
    --output_dir path/to/finetuned/model \
    --do_predict \
    --predict_with_generate \
    --model_type <pegasus|layout-pegasus|bigbird-pegasus|layout-bigbird-pegasus|mbart|layout-mbart|bigbird-mbart|layout-bigbird-mbart> \
    --model_name_or_path path/to/finetuned/model \
    --use_fast_tokenizer \
    --dataset_name <arxiv_lay|pubmed_lay|hal|scielo_es|scielo_pt|koreascience> \
    --data_dir path/to/data/dir \
    --max_source_length <1024|3072|4096> \
    --max_target_length 256 \
    --pad_to_max_length \
    --num_beams <5|8> \
    --length_penalty 0.8 
~~~
