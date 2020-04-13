# domain-trained-embeddings
# DENSE
python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/mimicked_Google_400k.bin --target-class toxic > ./results/1_toxic_DENSE_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/mimicked_Google_400k.bin --target-class severe_toxic > ./results/2_severe_toxic_DENSE_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/mimicked_Google_400k.bin --target-class obscene > ./results/3_obscene_DENSE_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/mimicked_Google_400k.bin --target-class threat > ./results/4_threat_DENSE_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/mimicked_Google_400k.bin --target-class identity_hate > ./results/5_identity_hate_DENSE_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/mimicked_Google_400k.bin --target-class insult > ./results/6_insult_DENSE_mimicked.log
sleep 360

# CNN
python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/mimicked_Google_400k.bin --target-class toxic > ./results/1_toxic_CNN_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/mimicked_Google_400k.bin --target-class severe_toxic > ./results/2_severe_toxic_CNN_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/mimicked_Google_400k.bin --target-class obscene > ./results/3_obscene_CNN_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/mimicked_Google_400k.bin --target-class threat > ./results/4_threat_CNN_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/mimicked_Google_400k.bin --target-class identity_hate > ./results/5_identity_hate_CNN_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/mimicked_Google_400k.bin --target-class insult > ./results/6_insult_CNN_mimicked.log
sleep 360


# LSTM
python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/mimicked_Google_400k.bin --target-class toxic > ./results/1_toxic_LSTM_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/mimicked_Google_400k.bin --target-class severe_toxic > ./results/2_severe_toxic_LSTM_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/mimicked_Google_400k.bin --target-class obscene > ./results/3_obscene_LSTM_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/mimicked_Google_400k.bin --target-class threat > ./results/4_threat_LSTM_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/mimicked_Google_400k.bin --target-class identity_hate > ./results/5_identity_hate_LSTM_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/mimicked_Google_400k.bin --target-class insult > ./results/6_insult_LSTM_mimicked.log
sleep 360


# LSTM + ATTENTION
python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/mimicked_Google_400k.bin --target-class toxic > ./results/1_toxic_LSTM_ATTENTION_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/mimicked_Google_400k.bin --target-class severe_toxic > ./results/2_severe_toxic_LSTM_ATTENTION_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/mimicked_Google_400k.bin --target-class obscene > ./results/3_obscene_LSTM_ATTENTION_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/mimicked_Google_400k.bin --target-class threat > ./results/4_threat_LSTM_ATTENTION_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/mimicked_Google_400k.bin --target-class identity_hate > ./results/5_identity_hate_LSTM_ATTENTION_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/mimicked_Google_400k.bin --target-class insult > ./results/6_insult_LSTM_ATTENTION_mimicked.log
sleep 360
