export decay=0.85
python test_prompt.py \
--model_path checkpoints/ \
--dataset dataset/validation_demo.jsonl \
--save_path output/test \
--split validation \
--batch_size 1

