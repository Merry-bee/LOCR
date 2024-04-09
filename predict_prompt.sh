export decay=0.85
python predict_prompt.py data/demo.pdf \
--out output/demo \
--checkpoint checkpoints/ \
--batchsize 4 \
--cuda "cuda:0" \
--recompute \
--return_attention True
