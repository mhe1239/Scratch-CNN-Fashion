cd "C:\src(3-1)\ArtificialIntelligence\w99"
python -u NoTrainerTrain.py *>&1 | Out-File -FilePath "C:\src(3-1)\ArtificialIntelligence\w99\training.log" -Encoding utf8 -Append
