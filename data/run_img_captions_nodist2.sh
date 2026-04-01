export CUDA_VISIBLE_DEVICES=2
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "e-cigarettes"
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "hookah"
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "gum"