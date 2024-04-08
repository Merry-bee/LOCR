export decay=0.85
python predict_prompt.py /mnt/petrelfs/share_data/zhonghansen/llm4science/ocrdataset/arxiv_train_data/repet_100/arxiv/1010.3213.pdf \
--out output/predict \
--checkpoint /mnt/petrelfs/share_data/zhonghansen/llm4science/ocrdataset/20240309/ \
--ckpt_path /mnt/petrelfs/share_data/zhonghansen/llm4science/ocrdataset/20240309/last.ckpt \
--batchsize 4 \
--cuda "cuda:0" \
--recompute \
--return_attention True
