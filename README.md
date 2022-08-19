# <div align="center"> PEERRec: An AI-based approach to Automatically Generate Recommendations and Predict Decisions in Peer Review</div>
This repository contains dataset and code of the "PEERRec: An AI-based approach to Automatically Generate Recommendations and Predict Decisions in Peer Review" Authors: Prabhat Kumar Bharti, Tirthankar Ghoshal, Mayank Agrawal, Asif Ekbal, Affiliation: Indian Institute of Technology, Patna, India

# Preprocessing dataset.

Scrape ICLR reviews and papers from Openreview, convert PDF of papers to JSON using Scienceparse library, and rename combine them in a folder in the following format:

reviews.json  : Corresponding to reviews of a paper

reviews.paper.json : Paper corresponding to the review above

Further, follow the following steps:
## 1. JSON to CSV
```
python ./Preprocessing/review_paper_json_to_csv.py
```

## 2. Create Section wise summary of papers
```
python ./Preprocessing/Create_paper__sections_summary.py \
--papers_pdf path_to_CSV_of_papers_created_in_step_1
```

## 3. Split data in row-wise instances
```
python ./Preprocessing/splitting.py
```

```
python ./Preprocessing/SplittingPaper.py
```

```
python ./Preprocessing/Create_sentencewise_files.py \
--dataset path_to_directory_of_above_output_files
```

## 4. Create Embeddings

```
python ./Preprocessing/Create_review_embeddings.py  \
--dataset path_to_directory_of_above_output_files
```

```
python ./Preprocessing/Create_paper_sections_embeddings.py \
--dataset path_to_directory_of_above_output_files
```

```
python ./Preprocessing/Create_VADER_sentiment_matrix.py \
--dataset path_to_directory_of_above_output_files
```

We provide the Preprocessed database here for ICLR [2017](https://drive.google.com/drive/folders/1xw8m0F6nvpd7Xf4Jfoxsg5N30CwQXY-P?usp=sharing), [2018](https://drive.google.com/drive/folders/1rIe2r2hxPrOGVl5Fb-lQbJIEna2E8snl?usp=sharing), [2019](https://drive.google.com/drive/folders/1SqtiZCqeiJK5OwP3jZiCT6Ftje8vqcP4?usp=sharing), [2020](https://drive.google.com/drive/folders/1JMY7Cys6BvA0Qn1AjLd54ni7JyAr4nU2?usp=sharing) and [2021](https://drive.google.com/drive/folders/1UugAjp43p6tHZReNSz2LIkpWOBUUygHI?usp=sharing).


# Proposed Model

## For running our proposed model for Recommendation Score prediction, run:
```
python ./regression_peer_review.py   \
--dataset path_to_preprocessed_files_directory
```

## For running our proposed model for Acceptance prediction, run:
```
python ./classification_peer_review.py    \
--dataset path_to_preprocessed_files_directory
```
