cd "C:\Scratch-CNN-Fashion\model"
python -u flexconvnet_pt_id6.py *>&1 | Out-File -FilePath "C:\Scratch-CNN-Fashion\model\training.log" -Encoding utf8 -Append
