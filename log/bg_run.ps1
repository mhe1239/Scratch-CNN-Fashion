cd "C:\Scratch-CNN-Fashion\model"
python -u NoTrainerTrain-Test.py *>&1 | Out-File -FilePath "C:\Scratch-CNN-Fashion\log\training.log" -Encoding utf8 -Append
