# News Headline Classification

## Summary
In this project, we tackled the task of classifying political news headlines by source — either \textit{Fox News} or \textit{NBC News}. This short-text binary classification problem has broader implications in \textbf{media bias detection}, \textbf{source attribution}, and \textbf{political analysis}. Our dataset contained over \textbf{3,800 headlines}, each labeled by origin. 

We began by evaluating which \textbf{normalization technique} (cleaning, stemming, or lemmatization) performed best for both \textbf{TF-IDF-based} and \textbf{RoBERTa-based} approaches. For TF-IDF, \textbf{stemming} yielded the best results, while for RoBERTa, \textbf{lemmatization} was superior. 

Next, we compared \textbf{contextual embeddings from RoBERTa} against TF-IDF vectors. Across all classical models, \textbf{RoBERTa embeddings outperformed TF-IDF}, showing improved representation quality. 

We then experimented with multiple \textbf{deep learning architectures} — LSTMs, Bi-GRUs, CNNs, and ANNs — including hybrid combinations. However, the best results were achieved with the standalone \textbf{Bi-LSTM} model. 

Finally, we fine-tuned the \textbf{entire RoBERTa + Bi-LSTM pipeline} end-to-end. This model achieved our best performance: \textbf{82.26\% accuracy} and a \textbf{macro F1-score of 0.82}, significantly improving over the \textbf{TF-IDF + Logistic Regression baseline of 70–71\%}. Thus, it was chosen as our \textbf{final model for deployment}.
