cd "C:\Scratch-CNN-Fashion\model"
python -u flexconvnet_train_pt.py *>&1 | Out-File -FilePath "C:\Scratch-CNN-Fashion\model\training.log" -Encoding utf8 -Append
shutdown /s /f /t 60
