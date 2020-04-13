# domain-trained-embeddings
# DENSE
python deep_learning_training_and_test_BERT.py --model 1 --target-class toxic > ./results/1_toxic_DENSE_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 1 --target-class severe_toxic > ./results/2_severe_toxic_DENSE_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 1 --target-class obscene > ./results/3_obscene_DENSE_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 1 --target-class threat > ./results/4_threat_DENSE_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 1 --target-class identity_hate > ./results/5_identity_hate_DENSE_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 1 --target-class insult > ./results/6_insult_DENSE_bert.log
sleep 360


#CNN
python deep_learning_training_and_test_BERT.py --model 2 --target-class toxic > ./results/1_toxic_CNN_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 2 --target-class severe_toxic > ./results/2_severe_toxic_CNN_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 2 --target-class obscene > ./results/3_obscene_CNN_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 2 --target-class threat > ./results/4_threat_CNN_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 2 --target-class identity_hate > ./results/5_identity_hate_CNN_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 2 --target-class insult > ./results/6_insult_CNN_bert.log
sleep 360



#LSTM
python deep_learning_training_and_test_BERT.py --model 3 --target-class toxic > ./results/1_toxic_LSTM_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 3 --target-class severe_toxic > ./results/2_severe_toxic_LSTM_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 3 --target-class obscene > ./results/3_obscene_LSTM_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 3 --target-class threat > ./results/4_threat_LSTM_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 3 --target-class identity_hate > ./results/5_identity_hate_LSTM_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 3 --target-class insult > ./results/6_insult_LSTM_bert.log
sleep 360



#LSTM + ATTENTION
python deep_learning_training_and_test_BERT.py --model 4 --target-class toxic > ./results/1_toxic_BIDIRECTIONAL_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 4 --target-class severe_toxic > ./results/2_severe_toxic_BIDIRECTIONAL_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 4 --target-class obscene > ./results/3_obscene_BIDIRECTIONAL_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 4 --target-class threat > ./results/4_threat_BIDIRECTIONAL_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 4 --target-class identity_hate > ./results/5_identity_hate_BIDIRECTIONAL_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 4 --target-class insult > ./results/6_insult_BIDIRECTIONAL_bert.log
sleep 360



