# domain-trained-embeddings
# DENSE
python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class toxic > ./results/1_toxic_DENSE_domain.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class severe_toxic > ./results/2_severe_toxic_DENSE_domain.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class obscene > ./results/3_obscene_DENSE_domain.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class threat > ./results/4_threat_DENSE_domain.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class identity_hate > ./results/5_identity_hate_DENSE_domain.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class insult > ./results/6_insult_DENSE_domain.log
sleep 360

# CNN
python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class toxic > ./results/1_toxic_CNN_domain.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class severe_toxic > ./results/2_severe_toxic_CNN_domain.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class obscene > ./results/3_obscene_CNN_domain.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class threat > ./results/4_threat_CNN_domain.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class identity_hate > ./results/5_identity_hate_CNN_domain.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class insult > ./results/6_insult_CNN_domain.log
sleep 360


# LSTM
python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class toxic > ./results/1_toxic_LSTM_domain.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class severe_toxic > ./results/2_severe_toxic_LSTM_domain.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class obscene > ./results/3_obscene_LSTM_domain.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class threat > ./results/4_threat_LSTM_domain.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class identity_hate > ./results/5_identity_hate_LSTM_domain.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class insult > ./results/6_insult_LSTM_domain.log
sleep 360


# LSTM + ATTENTION
python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class toxic > ./results/1_toxic_LSTM_ATTENTION_domain.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class severe_toxic > ./results/2_severe_toxic_LSTM_ATTENTION_domain.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class obscene > ./results/3_obscene_LSTM_ATTENTION_domain.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class threat > ./results/4_threat_LSTM_ATTENTION_domain.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class identity_hate > ./results/5_identity_hate_LSTM_ATTENTION_domain.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class insult > ./results/6_insult_LSTM_ATTENTION_domain.log
sleep 360
