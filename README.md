# Movie-Review Sentiment Analysis with an LSTM

A deep-learning model that reads a movie review and predicts whether it is **positive or negative**, served through a small **Streamlit web app**.

**Result:** 88.2% accuracy on 10,000 held-out reviews.

---

## The problem

Businesses collect far more written feedback (reviews, support tickets, survey comments) than anyone can read. A model that reliably tags sentiment lets a team track satisfaction over time and spot unhappy customers early. This project builds that capability on a standard benchmark: 50,000 IMDB movie reviews.

## Data

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle), balanced between positive and negative. The data is not included in this repo.

## Approach

1. **Cleaned the text:** removed HTML tags (e.g. `<br />`), punctuation and numbers, and lowercased everything.
2. **Turned words into numbers:** a tokenizer keeps the 5,000 most frequent words, and each review is padded or trimmed to 200 tokens.
3. **Split** the data 80/20 into training and test sets.
4. **Built the network:**
   - an **embedding layer** (128 dimensions) that learns a vector for each word;
   - an **LSTM layer** (64 units) that reads the review in order, so context and negation ("not good") can count;
   - a **sigmoid output** giving the probability that the review is positive.
5. **Trained** for 5 epochs (batch size 64) with the Adam optimiser and binary cross-entropy loss.
6. **Deployed:** saved the model and tokenizer, then wrapped them in a Streamlit app.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **88.2%** |
| Training accuracy (final epoch) | 94.2% |

![Training vs validation accuracy](images/training_accuracy.png)

Validation accuracy peaked at about 88.7% around epochs 3–4. After that, training accuracy kept rising while validation loss increased (0.28 → 0.34). The model was starting to **overfit**, so 3 epochs would have been enough.

## The app

`app.py` loads the trained model and lets anyone type a review and get a prediction with a confidence score.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects `imdb_lstm_model.h5` and `tokenizer.pkl` in the same folder. Running the notebook creates both.

## What I'd do next

- **Separate validation set and early stopping.** Here the test set was also used to monitor training. A proper three-way split, with training stopped when validation loss rises, would give a cleaner accuracy estimate and prevent the overfitting above.
- **Compare against a pretrained transformer** (e.g. DistilBERT) on the same test set, to see how much accuracy the extra model size buys.
- **Error analysis:** read the misclassified reviews. Sarcasm and mixed reviews ("great acting, terrible plot") are the likely failure cases.
- **Host the app** on Streamlit Community Cloud so it can be tried from a link.

## Tools

Python · TensorFlow/Keras · pandas · scikit-learn · Matplotlib · Streamlit · Google Colab

---

*Built as part of my MSc Data Science at KCA University, then written up as a standalone project.*
