# News Headline Classification

## Summary
In this project, we tackled the task of classifying political news headlines by source — either Fox News or
NBC News. This short-text binary classification problem has broader implications in media bias detection,
source attribution, and political analysis. Our dataset contained over 3,800 headlines, each labeled
by origin.
We began by evaluating which normalization technique (cleaning, stemming, or lemmatization) performed
best for both TF-IDF-based and RoBERTa-based approaches. For TF-IDF, stemming yielded the best
results, while for RoBERTa, lemmatization was superior.
Next, we compared contextual embeddings from RoBERTa against TF-IDF vectors. Across all clas-
sical models, RoBERTa embeddings outperformed TF-IDF, showing improved representation qual-
ity.
We then experimented with multiple deep learning architectures — LSTMs, Bi-GRUs, CNNs, and ANNs
— including hybrid combinations. However, the best results were achieved with the standalone Bi-LSTM
model.
Finally, we fine-tuned the entire RoBERTa + Bi-LSTM pipeline end-to-end. This model achieved our
best performance: 82.26% accuracy and a macro F1-score of 0.82, significantly improving over the
TF-IDF + Logistic Regression baseline of 70–71%. Thus, it was chosen as our final model for
deployment.
