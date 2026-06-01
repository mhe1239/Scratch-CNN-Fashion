cd "C:\pytorch_src\model\"
python -u flexconvnet_pt.py *>&1 | Out-File -FilePath "C:\Scratch-CNN-Fashion\model\training.log" -Encoding utf8 -Append
