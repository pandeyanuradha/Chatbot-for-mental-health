from typing imprt List
import unittest

import ddt
from ddt import unpack, data
import numpy as np
import pandas as pd
from transformers import OpenAIGPTTokenizer

from utils import sample_candidates, convert_df_to_conv_ai_dict


# fmt: off
class UtilsTest(unittest.TestCase):

    def test_sample_candidates(self):
        # Test that split id doesn't show up
        # Create random DF
        df = pd.DataFrame(data, columns=["questionID", "answerText", 'split'])

        for i in range(5):
            candidates = sample_candidates(df, , "questionID", "answerText", n_candidates)
            # Check that the samples don't come from the true data

    def test_fuzz_convert_df_to_conv_ai_dict(self):
        df = pd.read_csv("data/20200325_counsel_chat.csv")
        df = df[df["split"] == "train"]
        tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        for i in range(5):
            temp_df = df.sample(100)
            max_tokens = np.random.randint(1, 200)
            n_candidates = np.random.randint(1, 10)
            d = convert_df_to_conv_ai_dict(temp_df,
                                           [""],
                                           ["answerText"],
                                           tokenizer,
                                           max_tokens=max_tokens,
                                           n_candidates=n_candidates)

            # Test max length
            self.assertLessEqual(max([len(x["utterances"][0]["history"][0].split()) for x in d["train"]]), max_tokens)

            # Test n_candidates is equal to the number in the candidates list plus the one true response.
            train_lengths = [len(x["utterances"][0]["candidates"]) for x in d["train"]]
            self.assertEqual(n_candidates + 1, max(train_lengths))
            self.assertEqual(n_candidates + 1, min(train_lengths))


if __name__ == "__main__":
    unittest.main()
