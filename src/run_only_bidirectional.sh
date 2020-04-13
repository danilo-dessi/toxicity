# BIDIRECTIONAL pre trained
python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/GoogleNews-vectors-negative300.bin --target-class toxic > ./results/1_model5_toxic_BIDIRECTIONAL_google_pretrained.log
sleep 36

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/GoogleNews-vectors-negative300.bin --target-class severe_toxic > ./results/2_model5_severe_toxic_BIDIRECTIONAL_google_pretrained.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/GoogleNews-vectors-negative300.bin --target-class obscene > ./results/3_model5_obscene_BIDIRECTIONAL_google_pretrained.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/GoogleNews-vectors-negative300.bin --target-class threat > ./results/4_model5_threat_BIDIRECTIONAL_google_pretrained.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/GoogleNews-vectors-negative300.bin --target-class identity_hate > ./results/5_model5_identity_hate_BIDIRECTIONAL_google_pretrained.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/GoogleNews-vectors-negative300.bin --target-class insult > ./results/6_model5_insult_BIDIRECTIONAL_google_pretrained.log
sleep 360



# BIDIRECTIONAL mimicked

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/mimicked_Google_400k.bin --target-class toxic > ./results/1_model5_toxic_BIDIRECTIONAL_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/mimicked_Google_400k.bin --target-class severe_toxic > ./results/2_model5_severe_toxic_BIDIRECTIONAL_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/mimicked_Google_400k.bin --target-class obscene > ./results/3_model5_obscene_BIDIRECTIONAL_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/mimicked_Google_400k.bin --target-class threat > ./results/4_model5_threat_BIDIRECTIONAL_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/mimicked_Google_400k.bin --target-class identity_hate > ./results/5_model5_identity_hate_BIDIRECTIONAL_mimicked.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/mimicked_Google_400k.bin --target-class insult > ./results/6_model5_insult_BIDIRECTIONAL_mimicked.log
sleep 360



# BIDIRECTIONAL domain
python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/word2vec_toxic_300.bin --target-class toxic > ./results/1_model5_toxic_BIDIRECTIONAL_domain.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/word2vec_toxic_300.bin --target-class severe_toxic > ./results/2_model5_severe_toxic_BIDIRECTIONAL_domain.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/word2vec_toxic_300.bin --target-class obscene > ./results/3_model5_obscene_BIDIRECTIONAL_domain.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/word2vec_toxic_300.bin --target-class threat > ./results/4_model5_threat_BIDIRECTIONAL_domain.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/word2vec_toxic_300.bin --target-class identity_hate > ./results/5_model5_identity_hate_BIDIRECTIONAL_domain.log
sleep 360

python deep_learning_training_and_test.py --model 5 --emb-file ../../resources/word2vec_toxic_300.bin --target-class insult > ./results/6_model5_insult_BIDIRECTIONAL_domain.log
sleep 360