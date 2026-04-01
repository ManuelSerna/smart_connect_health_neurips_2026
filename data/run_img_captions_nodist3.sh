export CUDA_VISIBLE_DEVICES=3
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "lozenges"
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "cigarettes"
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "smokeless_tobacco"
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "uncategorized"

