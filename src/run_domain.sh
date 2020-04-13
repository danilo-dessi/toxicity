# domain-trained-embeddings
# DENSE
python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class toxic > ./results/1_model1_toxic_DENSE_domain.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class severe_toxic > ./results/2_model1_severe_toxic_DENSE_domain.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class obscene > ./results/3_model1_obscene_DENSE_domain.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class threat > ./results/4_model1_threat_DENSE_domain.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class identity_hate > ./results/5_model1_identity_hate_DENSE_domain.log
sleep 360

python deep_learning_training_and_test.py --model 1 --emb-file ../../resources/word2vec_toxic_300.bin --target-class insult > ./results/6_model1_insult_DENSE_domain.log
sleep 360

# CNN
python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class toxic > ./results/1_model2_toxic_CNN_domain.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class severe_toxic > ./results/2_model2_severe_toxic_CNN_domain.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class obscene > ./results/3_model2_obscene_CNN_domain.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class threat > ./results/4_model2_threat_CNN_domain.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class identity_hate > ./results/5_model2_identity_hate_CNN_domain.log
sleep 360

python deep_learning_training_and_test.py --model 2 --emb-file ../../resources/word2vec_toxic_300.bin --target-class insult > ./results/6_model2_insult_CNN_domain.log
sleep 360


# LSTM
python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class toxic > ./results/1_model3_toxic_LSTM_domain.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class severe_toxic > ./results/2_model3_severe_toxic_LSTM_domain.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class obscene > ./results/3_model3_obscene_LSTM_domain.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class threat > ./results/4_model3_threat_LSTM_domain.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class identity_hate > ./results/5_model3_identity_hate_LSTM_domain.log
sleep 360

python deep_learning_training_and_test.py --model 3 --emb-file ../../resources/word2vec_toxic_300.bin --target-class insult > ./results/6_model3_insult_LSTM_domain.log
sleep 360


# LSTM + ATTENTION
python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class toxic > ./results/1_model4_toxic_LSTM_ATTENTION_domain.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class severe_toxic > ./results/2_model4_severe_toxic_LSTM_ATTENTION_domain.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class obscene > ./results/3_model4_obscene_LSTM_ATTENTION_domain.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class threat > ./results/4_model4_threat_LSTM_ATTENTION_domain.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class identity_hate > ./results/5_model4_identity_hate_LSTM_ATTENTION_domain.log
sleep 360

python deep_learning_training_and_test.py --model 4 --emb-file ../../resources/word2vec_toxic_300.bin --target-class insult > ./results/6_model4_insult_LSTM_ATTENTION_domain.log
sleep 360
