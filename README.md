# News Headline Classification

## Summary
In this project, we tackled the task of classifying political news headlines by source — either Fox News or
NBC News. This short-text binary classification problem has broader implications in **media bias detection,
source attribution**, and political analysis. Our dataset contained over **3,800 headlines**, each labeled
by origin.

We began by evaluating which **normalization technique** (cleaning, stemming, or lemmatization) performed
best for both **TF-IDF-based** and **RoBERTa-based** approaches. For TF-IDF, **stemming** yielded the best
results, while for RoBERTa, **lemmatization** was superior.
Next, we compared **contextual embeddings from RoBERTa** against TF-IDF vectors. Across all clas-
sical models, **RoBERTa embeddings outperformed TF-IDF**, showing improved representation qual-
ity.

We then experimented with multiple **deep learning architectures** — LSTMs, Bi-GRUs, CNNs, and ANNs
— including hybrid combinations. However, the best results were achieved with the standalone **Bi-LSTM**
model.

Finally, we fine-tuned the **entire RoBERTa + Bi-LSTM pipeline** end-to-end. This model achieved our
best performance: **82.26% accuracy** and a **macro F1-score of 0.82**, significantly improving over the
**TF-IDF + Logistic Regression baseline of 70–71%**. Thus, it was chosen as our **final model for
deployment.**

## Code File Description
* 1_Dhruv_Kruthi_AML_Web_Scraping_Final.ipynb - Performed Web Scraping
* 2_New_Dhruv_Kruthi_Normalization.ipynb - Cleaned and Normalized text to Stemmed, Lem-
matized, and Cleaned
* 3_Dhruv_Kruthi_RoBERTa_Stem.ipynb - Computed RoBERTa embeddings for Stemmed Text
* 4_Dhruv_Kruthi_RoBERTa_Lemma.ipynb - Computed RoBERTa embeddings for Lemmatized
Text
* 5_Dhruv_Kruthi_RoBERTa_Clean.ipynb - Computed RoBERTa embeddings for Cleaned Text
* 6_Dhruv_Kruthi_explore1.ipynb - Answered Exploratory Question 1
* 7_Dhruv_Kruthi_explore2.ipynb - Answered Exploratory Question 2
* 8_Dhruv_Kruthi_Final_RoBERTa_lemma.ipynb - Answered Exploratory Question 3
* 9_Dhruv_Kruthi_RoBERTa_lemma_tuning_e10.ipynb - Answered Exploratory Question 4
* 10_Dhruv_Kruthi_Hidden_Test.ipynb - Final Predictions on Hidden Test Data
