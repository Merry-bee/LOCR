export decay=0.85
python test_prompt.py \
--model_path checkpoints/ \
--ckpt_path checkpoints/last.ckpt \
--dataset /mnt/petrelfs/share_data/zhonghansen/llm4science/ocrdataset/arxiv_train_data/good/validation1000_0306.jsonl \
--save_path output/arxiv_decay_085 \
--split validation \
--batch_size 1

