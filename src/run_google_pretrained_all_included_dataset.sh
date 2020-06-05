# domain-trained-embeddings
# DENSE
python deep_learning_training_and_test_all_included_dataset.py --model 1 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class toxic > ./results_all_included/1_toxic_DENSE_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 1 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class severe_toxic > ./results_all_included/2_severe_toxic_DENSE_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 1 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class obscene > ./results_all_included/3_obscene_DENSE_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 1 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class threat > ./results_all_included/4_threat_DENSE_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 1 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class identity_hate > ./results_all_included/5_identity_hate_DENSE_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 1 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class insult > ./results_all_included/6_insult_DENSE_google_pretrained.log
sleep 360

# CNN
python deep_learning_training_and_test_all_included_dataset.py --model 2 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class toxic > ./results_all_included/1_toxic_CNN_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 2 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class severe_toxic > ./results_all_included/2_severe_toxic_CNN_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 2 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class obscene > ./results_all_included/3_obscene_CNN_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 2 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class threat > ./results_all_included/4_threat_CNN_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 2 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class identity_hate > ./results_all_included/5_identity_hate_CNN_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 2 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class insult > ./results_all_included/6_insult_CNN_google_pretrained.log
sleep 360


# LSTM
python deep_learning_training_and_test_all_included_dataset.py --model 3 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class toxic > ./results_all_included/1_toxic_LSTM_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 3 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class severe_toxic > ./results_all_included/2_severe_toxic_LSTM_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 3 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class obscene > ./results_all_included/3_obscene_LSTM_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 3 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class threat > ./results_all_included/4_threat_LSTM_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 3 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class identity_hate > ./results_all_included/5_identity_hate_LSTM_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 3 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class insult > ./results_all_included/6_insult_LSTM_google_pretrained.log
sleep 360


# LSTM + ATTENTION
python deep_learning_training_and_test_all_included_dataset.py --model 4 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class toxic > ./results_all_included/1_toxic_LSTM_ATTENTION_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 4 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class severe_toxic > ./results_all_included/2_severe_toxic_LSTM_ATTENTION_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 4 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class obscene > ./results_all_included/3_obscene_LSTM_ATTENTION_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 4 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class threat > ./results_all_included/4_threat_LSTM_ATTENTION_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 4 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class identity_hate > ./results_all_included/5_identity_hate_LSTM_ATTENTION_google_pretrained.log
sleep 360

python deep_learning_training_and_test_all_included_dataset.py --model 4 --emb-file ../resources/GoogleNews-vectors-negative300.bin --target-class insult > ./results_all_included/6_insult_LSTM_ATTENTION_google_pretrained.log
sleep 360
