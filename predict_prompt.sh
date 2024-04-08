export decay=0.85
python predict_prompt.py /mnt/petrelfs/share_data/zhonghansen/llm4science/ocrdataset/arxiv_train_data/repet_100/arxiv/2301.05072.pdf \
--out output/predict \
--checkpoint checkpoints/ \
--ckpt_path checkpoints/last.ckpt \
--batchsize 4 \
--cuda "cuda:0" \
--recompute \
--return_attention True
