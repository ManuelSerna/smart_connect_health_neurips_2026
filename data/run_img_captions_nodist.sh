export CUDA_VISIBLE_DEVICES=1
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "cigars"
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "heated_tobacco"
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "patches"
python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "pipe_tobacco"
