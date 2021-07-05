from typing import Dict, Any, Callable, List, Tuple, Optional, Union
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, TransformerMixin
import re


def calculate_classification_metrics(
    y_true: np.array,
    y_pred: np.array,
    average: Optional[str] = None,
    return_df: bool = True,
) -> Union[Dict[str, float], pd.DataFrame]:
    """Computes f1, precision, recall, kappa, accuracy, and support

    Args:
        y_true: The true labels
        y_pred: The predicted labels
        average: How to average multiclass results
        return_df: Returns a dataframe if true otherwise a dictionary of performance
            values.

    Returns:
        Either a dataframe of the performance metrics or a single dictionary
    """
    labels = unique_labels(y_true, y_pred)

    # get results
    precision, recall, f_score, support = metrics.precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average
    )

    kappa = metrics.cohen_kappa_score(y_true, y_pred, labels=labels)
    accuracy = metrics.accuracy_score(y_true, y_pred)

    # create a pandas DataFrame
    if return_df:
        results = pd.DataFrame(
            {
                "class": labels,
                "f_score": f_score,
                "precision": precision,
                "recall": recall,
                "support": support,
                "kappa": kappa,
                "accuracy": accuracy,
            }
        )
    else:
        results = {
            "f1": f_score,
            "precision": precision,
            "recall": recall,
            "kappa": kappa,
            "accuracy": accuracy,
        }

    return results


def visualize_performance(
        df: pd.DataFrame,
        metrics: List[str],
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        use_class_names: bool = True
) -> None:
    """Takes a Performance DF and converts it to a bar plot performance graph

    Args:
        df: A dataframe where each row is a class and each column is a metric
        metrics: A list of metrics from the columns of df to plot
        ax: A matplotlib axes object that we want to draw the plot on
        title: The title of the plot
        ylim: The minimum and maximum range for the yaxis.
        figsize: The width and height of the figure.  This does nothing if ax is set
        use_class_names: This will label the x ticks with the class name in a multiclass setting.
    """
    unstacked_df = (
        df[metrics]
            .T.unstack()
            .reset_index()
            .rename(
            index=str, columns={"level_0": "class", "level_1": "metric", 0: "score"}
        )
    )

    if use_class_names:
        unstacked_df["class"] = unstacked_df["class"].apply(
            lambda x: df["class"].tolist()[x]
        )

    if figsize is None:
        figsize = (10, 7)

    # Diplay the graph
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    sns.barplot(x="class", y="score", hue="metric", data=unstacked_df, ax=ax)

    # Format the graph
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    if title is not None:
        ax.set_title(title, fontsize=20)

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()


class BertTransformer(BaseEstimator, TransformerMixin):
    """See https://towardsdatascience.com/build-a-bert-sci-kit-transformer-59d60ddd54a5#d608"""
    def __init__(
            self,
            bert_tokenizer,
            bert_model,
            max_length: int = 60,
            embedding_func: Optional[Callable[[Tuple[torch.tensor]], torch.tensor]] = None,
    ):
        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.model.eval()
        self.max_length = max_length
        self.embedding_func = embedding_func

        if self.embedding_func is None:
            self.embedding_func = lambda x: x[0][:, 0, :]

        # TODO:: PADDING

    def _tokenize(self, text: str):
        tokenized_text = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length
        )["input_ids"]
        attention_mask = [1] * len(tokenized_text)

        # bert takes in a batch so we need to unsqueeze the rows
        return (
            torch.tensor(tokenized_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text: str):
        tokenized, attention_mask = self._tokenize(text)

        embeddings = self.model(tokenized, attention_mask)
        return self.embedding_func(embeddings)

    def transform(self, text: List[str]):
        if isinstance(text, pd.Series):
            text = text.tolist()

        with torch.no_grad():
            return torch.stack([self._tokenize_and_predict(string) for string in text])

    def fit(self, X, y=None):
        """No fitting necessary so we just return ourselves"""
        return self


def convert_df_to_conv_ai_dict(df: pd.DataFrame,
                               personality: List[str],
                               response_columns: List[str],
                               tokenizer: Callable[[str], List[str]],
                               max_tokens: Optional[int] = None,
                               n_candidates: int = 6
                               ) -> Dict[str, List[Any]]:
    """
    Each entry in personachat is a dict with two keys personality and utterances, the dataset is a list of entries.
    personality:  list of strings containing the personality of the agent
    utterances: list of dictionaries, each of which has two keys which are lists of strings.
        candidates: [next_utterance_candidate_1, ..., next_utterance_candidate_19]
            The last candidate is the ground truth response observed in the conversational data
        history: [dialog_turn_0, ... dialog_turn N], where N is an odd number since the other user starts every conversation.
    Preprocessing:
        - Spaces before periods at end of sentences
        - everything lowercase

    Process each row of a DataFrame.  For each row:
    1. Grab the conversational input text
    2. Grab A the responses
    3. Create a unique data entry for each response to the question.
    4. Sample random response sentences from the dataset.
    5. Combine the random responses into a candidate list.

    Args:
        df: The counsel chat pandas dataframe
        personality: The personality we would like to use during training
        response_columns: Columns which contain valid responses to the question.  For example,
            the answerText column is the complete response of the therapist
        tokenizer: The transformers library tokenizer associated with the model we will be
            training.  It is used for setting the maximum sequence length
        max_tokens: The maximum number of tokens that any candidate, response, or question should be.
        n_candidates: The number of candidate phrases to include in the dataset for training.
            The last member of candidates is the ground truth response

    Returns:
        A dictionary with a train and validation key.
    """
    # Add one because the index of the dataframe is the 0th position.
    tuple_map = {name: index + 1 for index, name in enumerate(df.columns.tolist())}

    train = []
    val = []
    # Step through every row in the dictionary
    for row in df.itertuples():

        # Get the question name and title
        # TODO:: MAKE THIS GENERAL YOU DUMB DUMB
        question_title = row[tuple_map["questionTitle"]]
        question_text = row[tuple_map["questionText"]]
        question_combined = question_title + " " + question_text

        # Step through every response column in the row
        for response_column in response_columns:

            # Get the true response
            true_response = row[tuple_map[response_column]]

            # We only want to add data if a good response exists
            if len(true_response) > 1:
                # Get candidate alternate sentances by sampling from all other questions
                candidates = sample_candidates(df, row[tuple_map["questionID"]], "questionID", "answerText",
                                               n_candidates)

                # Add the correct response to the end
                candidates.append(true_response)

                # We want to trim the size of the tokens
                if max_tokens is not None:
                    # Use the provided tokenizer to tokenize the input and truncate at max_tokens
                    question_combined = tokenizer.convert_tokens_to_string(
                        tokenizer.tokenize(question_combined)[:max_tokens])
                    candidates = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(candidate)[:max_tokens]) for
                                  candidate in candidates]

                if len(candidates) != n_candidates + 1:
                    print(true_response)
                    assert False

                # Define the personality and the history
                d = {"personality": personality,
                     "utterances": [{"history": [question_combined],
                                     "candidates": candidates}]}
                if getattr(row, "split") == "train":
                    train.append(d)
                elif getattr(row, "split") == "val":
                    val.append(d)

    data = {"train": train, "valid": val}

    return data


def sample_candidates(df: pd.DataFrame, current_id: Any, id_column: str, text_column: str, n: int) -> List[str]:
    """Samples candidate responses to a question from the dataframe

    It is aware of data splits and only samples from within the same split.  This avoids
    leaking information between training validation and testing.  The sampled responses are
    also drawn from all rows which do not have the same id as the current_id

    Args:
        df: The dataframe we want to sample responses from
        current_id: The unique identifier we would like to leave out of our sampling
        id_column: The column name in the dataframe with the unique ids.  current_id should
            be an element of this column
        text_column: The column with the text we want to sample
        n: How many samples we want to take.

    Returns:
        A list of samples strings from our dataframe.
    """
    # We must only sample candidates from the correct data split to avoid information leakage across channels
    split = df[df[id_column] == current_id]["split"].tolist()[0]
    candidate_df = df[df["split"] == split]

    # Sample 3 random rows from the dataframe not matching the current id
    sampled_texts = candidate_df[candidate_df[id_column] != current_id].sample(n + 15)[text_column].tolist()

    # join them all
    text = " ".join(sampled_texts)

    # Replace all newlines with spaces...
    text_no_newline = re.sub("\n", " ", text).lower()

    # Split on punctuation
    split_text = re.split('[?.!]', text_no_newline)

    # Remove all empty lines
    filtered_text = [x.strip() for x in split_text if len(x.strip()) > 1]

    # Shuffle the list
    return np.random.choice(filtered_text, n).tolist()