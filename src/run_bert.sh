# domain-trained-embeddings
# DENSE
python deep_learning_training_and_test_BERT.py --model 1 --target-class toxic > ./results/1_toxic_model1_DENSE_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 1 --target-class severe_toxic > ./results/2_severe_toxic_model1_DENSE_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 1 --target-class obscene > ./results/3_obscene_model1_DENSE_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 1 --target-class threat > ./results/4_threat_model1_DENSE_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 1 --target-class identity_hate > ./results/5_identity_hate_model1_DENSE_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 1 --target-class insult > ./results/6_insult_model1_DENSE_bert.log
sleep 360


#CNN
python deep_learning_training_and_test_BERT.py --model 2 --target-class toxic > ./results/1_toxic_model2_CNN_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 2 --target-class severe_toxic > ./results/2_severe_toxic_model2_CNN_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 2 --target-class obscene > ./results/3_obscene_model2_CNN_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 2 --target-class threat > ./results/4_threat_model2_CNN_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 2 --target-class identity_hate > ./results/5_identity_hate_model2_CNN_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 2 --target-class insult > ./results/6_insult_model2_CNN_bert.log
sleep 360



#LSTM
python deep_learning_training_and_test_BERT.py --model 3 --target-class toxic > ./results/1_toxic_model3_LSTM_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 3 --target-class severe_toxic > ./results/2_severe_toxic_model3_LSTM_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 3 --target-class obscene > ./results/3_obscene_model3_LSTM_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 3 --target-class threat > ./results/4_threat_model3_LSTM_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 3 --target-class identity_hate > ./results/5_identity_hate_model3_LSTM_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 3 --target-class insult > ./results/6_insult_model3_LSTM_bert.log
sleep 360



#LSTM + ATTENTION
python deep_learning_training_and_test_BERT.py --model 5 --target-class toxic > ./results/1_toxic_model5_BIDIRECTIONAL_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 5 --target-class severe_toxic > ./results/2_severe_toxic_model5_BIDIRECTIONAL_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 5 --target-class obscene > ./results/3_obscene_model5_BIDIRECTIONAL_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 5 --target-class threat > ./results/4_threat_model5_BIDIRECTIONAL_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 5 --target-class identity_hate > ./results/5_identity_hate_model5_BIDIRECTIONAL_bert.log
sleep 360

python deep_learning_training_and_test_BERT.py --model 5 --target-class insult > ./results/6_insult_model5_BIDIRECTIONAL123_bert.log
sleep 360



