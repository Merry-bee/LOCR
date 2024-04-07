export decay=0.85
python predict_prompt.py data/case/arxiv_case/1209.2821.pdf \
--out output/show_case \
--checkpoint /mnt/data/oss_beijing/sunyu/nougat/PromptNougat/result/nougat/20240309/ \
--ckpt_path /mnt/data/oss_beijing/sunyu/nougat/PromptNougat/result/nougat/20240309/last.ckpt \
--batchsize 4 \
--cuda "cuda:0" \
--recompute \
--return_attention True \
--interaction True
