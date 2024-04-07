export decay=0.75
python test_prompt.py \
--model_path /mnt/data/oss_beijing/sunyu/nougat/PromptNougat/result/nougat/20240207/ \
--ckpt_path /mnt/data/oss_beijing/sunyu/nougat/PromptNougat/result/nougat/20240207/last.ckpt \
--dataset data/arxiv_train_data/good_validation1000.jsonl \
--save_path output/arxiv_decay_075 \
--split validation \
--batch_size 1
# --visualize True \

