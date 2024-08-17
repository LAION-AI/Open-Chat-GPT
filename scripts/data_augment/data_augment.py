"""Script for a variety of data augmentation techniques for generating Question answer pairs.
Depending on the class used it takes in the input files and generates summaries from essays (which then will result in a "write a story about [summary]"-> essay pair),#
buggs code (in order to have bugged code + "please fix" -> code), ...
example usage:
  data_augment.py --dataset essays.tsv --augmenter hierarchicalsummarizer --output out.json
args:
  -- dataset: TSV file referencing txt files with essays/code
  -- augmenter: the augmenter used: one of 'essayinstruction', 'essayrevision', 'stackexchange', 'hierarchicalsummarizer', 'entityrecognizedsummarizer', 'codebugger"
  -- output: where to save the output
"""


import argparse
import json
import random
import string
from collections import Counter

import nltk
import pandas as pd
import requests
import spacy
import torch
from bs4 import BeautifulSoup as bs
from logic.logic_injector import LogicBug
from nltk.corpus import wordnet
from syntax.syntax_injector import SyntaxBug
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, pipeline


class DataAugmenter:
    def __init__(self):
        raise NotImplementedError()

    def parse(self, essays):
        prompts = []
        preds = []

        for essay in essays:
            essay_prompts, essay_preds = self.parse_single(essay)

            prompts += essay_prompts
            preds += essay_preds

        return prompts, preds

    def parse_single(self, essay):
        pass


class EssayInstructor(DataAugmenter):
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = "snrspeaks/t5-one-line-summary"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def parse_single(self, essay):
        essay_paragraphs = essay.split("\n\n")
        preds = []

        for para in essay_paragraphs:
            input_ids = self.tokenizer.encode(para, return_tensors="pt", add_special_tokens=True)
            generated_ids = self.model.generate(
                input_ids=input_ids,
                num_beams=5,
                max_length=35,
                repetition_penalty=4.5,
                length_penalty=1.5,
                early_stopping=True,
                num_return_sequences=1,
            )
            preds.append(
                self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            )

        prompts = (
            ["Write an intro paragraph to an essay called"]
            + ["Write a paragraph to an essay about"] * len(preds[1:-1])
            + ["Write a concluding paragraph about"]
        )

        assert len(preds) == len(prompts)
        prompts = [prompt + " " + pred for prompt, pred in zip(prompts, preds)]

        return prompts, essay_paragraphs


class EssayReviser(DataAugmenter):
    def __init__(self):
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    def parse_single(self, essay):
        instructions = []

        # Make structure error (shuffle one paragraph with another)
        essay_paragraphs = essay.split("\n\n")  # Splitting a String by newline character (\n)

        rand1 = random.randint(0, len(essay_paragraphs) - 1)
        rand2 = random.randint(0, len(essay_paragraphs) - 1)

        temp = essay_paragraphs[rand1]
        essay_paragraphs[rand1] = essay_paragraphs[rand2]
        essay_paragraphs[rand2] = temp

        corrupted_essay = "\n\n".join(essay_paragraphs)

        instructions.append("Fix structure errors in this essay" + corrupted_essay)

        essay_words = essay.split()
        for i in range(len(essay_words)):
            if random.randint(0, 100) < 30:
                suggestion = []
                for syn in wordnet.synsets(essay_words[i]):
                    for l in syn.lemmas():
                        suggestion.append(l.name())
                if suggestion != []:
                    essay_words[i] = suggestion[random.randint(0, len(suggestion) - 1)]

        corrupted_essay = " ".join(essay_words)

        instructions.append("Fix grammar errors in this essay: " + corrupted_essay)

        # you can change the number 60 to change how much corrupted this essay will be
        for _ in range(len(essay) // 60):
            rand = random.randint(0, len(essay))
            corrupted_essay = essay[:rand] + random.choice(string.ascii_letters) + essay[rand + 1 :]

        instructions.append("Fix typing errors in this essay" + corrupted_essay)

        return instructions, [essay] * len(instructions)


class StackExchangeBuilder(DataAugmenter):
    def __init__(self, base_url=None, filter_opts=None):
        self.base_url = (
            base_url
            if base_url is not None
            else "https://ia600107.us.archive.org/view_archive.php?archive=/27/items/stackexchange/{0}&file=Posts.xml"
        )
        self.filter_opts = (
            filter_opts if filter_opts is not None else ["accepted", "score", "convert_html", "clean_tags"]
        )

    def get_all_filenames(self):
        response = requests.get("https://archive.org/download/stackexchange")
        if response.ok:
            soup = bs(response.content, "html.parser")
            table = soup.find("table")
            link_tags = table.find_all("a")
            urls = {}
            for link in link_tags:
                url = link["href"]
                name = url.split(".stackexchange")[0].replace(".", "_").replace("-", "_")
                if url.endswith("7z"):
                    urls[name] = self.base_url.format(url)
            return urls

    def xml_to_df(self, response: str):
        """
        Collect and Manually import XML into Dataframe

        pd.read_xml() errors when XML trees are too large, this is just a hack to
        download a XML file and parse into a Dataframe. **Not Tested on huge XML files**

        Parameters:
        response (Requests.Response): Requests response object with the XML data

        Returns:
        df (DataFrame): A Dataframe from the XML file
        """
        xml_format_map = {
            "Id": int,
            "PostTypeId": int,
            "CreationDate": str,
            "Score": int,
            "ViewCount": int,
            "Body": str,
            "AnswerCount": int,
            "CommentCount": int,
            "ContentLicense": str,
            "AcceptedAnswerId": int,
            "ParentId": int,
        }
        soup = bs(response.content, "xml")
        posts = soup.find_all("row")

        all_posts = [post.attrs for post in posts]

        df = pd.DataFrame(all_posts)
        df.AnswerCount.fillna(0, inplace=True)
        df.ViewCount.fillna(0, inplace=True)
        df.AcceptedAnswerId.fillna(0, inplace=True)
        df.ParentId.fillna(0, inplace=True)
        df["DataSource"] = response.url
        df = df.astype(xml_format_map)
        return df

    def filter(self, df):
        if "accepted" in self.filter_opts:
            """**TODO**
            Filter only to Questions with Accepted Answers

            Filter dataframe by questions that have accepted answers, should also include
            all rows of answers for those questions, even if not accepted."""

            df = df[(df["AcceptedAnswerId"].notnull()) | (df["ParentId"] == df["Id"])]

        if "score" in self.filter_opts:
            """**TODO**
            Filter Dataframe by minimum scores

            Filter Question and Answer columns by score thresholds to trim lower scoring results"""
            question_score_threshold = 0
            answer_score_threshold = 5
            df = df[
                ((df["Score"] >= question_score_threshold) & (df.PostTypeId == 1))
                | ((df["Score"] >= answer_score_threshold) & (df.PostTypeId == 2))
            ]

        if "clean_tags" in self.filter_opts:
            """
            Convert Tags into Comma separated
            Converts Tag slugs into commas separated tags"""
            df["TagsClean"] = (
                df["Tags"].str.replace("-", " ").str.replace("><", ", ").str.replace("<", "").str.replace(">", "")
            )

        if "convert_html" in self.filter_opts:
            """
            Convert HTML tags to pure text

            Feeds HTML text body into BeautifulSoup to parse it to only text. Set aside as
            function to provide option to skip"""
            column = "Body"
            df.dropna(subset=[column], inplace=True)
            df[f"{column}Clean"] = df[column].apply(lambda row: bs(row, "html.parser").text)

        return df

    def parse(self, _):
        urls = self.get_all_filenames()
        dataset_name = "ai"

        xml_posts_path = urls.get(dataset_name)

        response = requests.get(xml_posts_path)
        df = self.xml_to_df(response)
        df = self.filter(df)

        questions = df[df.PostTypeId == 1]
        answers = df[df.PostTypeId == 2]

        df = pd.merge(
            questions,
            answers,
            left_on="Id",
            right_on="ParentId",
            suffixes=("_q", "_a"),
            how="left",
        )
        questions = df[["Title_q", "BodyClean_q"]]
        # prepend title to question and make questions to list
        questions = questions.apply(lambda x: x["Title_q"] + "\n" + x["BodyClean_q"], axis=1)
        questions = questions.tolist()

        answers = df[["BodyClean_a"]]
        answers = answers.tolist()

        return questions, answers


class HierachicalSummarizer(DataAugmenter):
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            "pszemraj/long-t5-tglobal-base-16384-book-summary",
            device=0 if torch.cuda.is_available() else -1,
        )

        self.params = {
            "max_length": 1024,
            "min_length": 8,
            "no_repeat_ngram_size": 3,
            "early_stopping": False,
            "repetition_penalty": 3.5,
            "length_penalty": 0.3,
            "encoder_no_repeat_ngram_size": 3,
            "num_beams": 4,
        }  # parameters for text generation out of model

        self.nlp = spacy.load("en_core_web_sm")

    def cleanup_summary(self, out):
        (
            out.replace("The novel begins with the description of", "")
            .replace("the description of", "")
            .replace("The novel begins", "")
            .replace("This chapter introduces us to", "")
            .replace("In this chapter, ", "")
            .replace("This chapter", "")
            .strip(" ,")
        )
        return out

    def parse_single(self, essay):
        final_summary = ""
        new_summary = ""
        level_2_summary = []
        level_1_summary = []
        entities = []
        essay_parts = essay.split("##")
        for section_text in essay_parts:
            result = self.summarizer(section_text, **self.params)
            out = self.cleanup_summary(result[0]["summary_text"])
            level_2_summary.append(out)
            result = self.summarizer(out, **self.params)
            out = self.cleanup_summary(result[0]["summary_text"])
            new_summary += "\n" + out
            level_1_summary.append(out)

            entity = recognize_entities(section_text, self.nlp, n=5, person="ignore")
            entities.append(entity)

        result = self.summarizer(new_summary, **self.params)
        final_summary = self.cleanup_summary(result[0]["summary_text"])

        first_instruction = "Write a story about the following:\n" + final_summary
        first_answer = "\n".join(level_1_summary)
        instructions = [first_instruction]
        answers = [first_answer]

        for entity, answer in zip(entities, level_2_summary):
            instructions.append(f"Now expand on {entity}!")
            answers.append(answer)

        for entity, answer in zip(entities, level_1_summary):
            instructions.append(f"Further expand on {entity}.")
            answers.append(answer)

        return instructions, answers


class EntityRecognizedSummarizer(DataAugmenter):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")  # run !python -m spacy download en_core_web_sm in order to download

    def parse_single(self, essay):
        ents = recognize_entities(essay, self.nlp)
        characters = ents.most_common(4, person=True)
        topic = recognize_entities(essay, self.nlp, n=2, person=False)

        question = f"Please write a story titled {topic} with the characters {characters}."
        answer = f"Sure. Here is a story titled {topic}\n" + essay

        return [question], [answer]


class CodeBugger(DataAugmenter):
    """
    https://github.com/LAION-AI/Open-Assistant/blob/main/notebooks/code-bugger/openbugger_example.md
    Openbugger is a Python package that allows you to inject syntax and logic errors into your code.
    This can be useful for testing the robustness of your code or for creating test cases for debugging exercises or for training an assistant to debug code.
    To install:
            cwd = os.getcwd()

        # Next, we'll use Git to clone the repository.
        subprocess.run(["git", "clone", "https://github.com/furlat/OpenBugger", cwd + "/OpenBugger"])

        # Now, we'll use pip to install the package from the local repository.
        subprocess.run(["python3", "-m", "pip", "install", "--editable", cwd + "/OpenBugger"])
    """

    def __init__(self):
        self.syntax_bug = SyntaxBug()
        self.logic_bug = LogicBug()

    def parse_single(self, code):
        code = self.syntax_bug(code, "medium", num_errors=2)
        code = self.logic_bug(code, "medium", num_errors=2)

        question = "Can you fix the following code?\n" + code

        answer = (
            "The following code is correct:\n"
            + code
            + "\nI hope I could help you fixing your code. In case you need more help, feel free to ask me again."
        )

        return [question], [answer]


class CodeInstructor(DataAugmenter):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Graverman/t5-code-summary")
        self.model = T5ForConditionalGeneration.from_pretrained("Graverman/t5-code-summary")

    def parse(self, codes):
        source_encoding = self.tokenizer(
            codes,
            max_length=300,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        outputs = self.model.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            max_length=100,
            length_penalty=0.75,
            repetition_penalty=2.5,
            early_stopping=True,
            use_cache=True,
        )
        summaries = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        questions = ["Write a script that does the following:\n" + s for s in summaries]
        answers = codes

        return questions, answers


def recognize_entities(text, model, n=4, person="ignore"):
    """Given a text and a model for entity recognition, return the most occurring entities in the text as a string"""
    doc = model(text)
    if person == "ignore":
        ents = Counter([ent.text.strip() for ent in list(doc.ents) if len(ent.text.strip()) >= 5])
    elif person:
        ents = Counter(
            [ent.text.strip() for ent in list(doc.ents) if ent.label_ == "PERSON" and len(ent.text.strip()) >= 5]
        )
    else:
        ents = Counter(
            [ent.text.strip() for ent in list(doc.ents) if ent.label_ != "PERSON" and len(ent.text.strip()) >= 5]
        )
    ents = ents.most_common(n)
    ents = ", ".join([a[0] for a in ents])

    return ents


def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--augmenter", type=str, required=True)
    args.add_argument("--output", type=str, required=True)
    args = args.parse_args()

    assert args.dataset.endswith(".tsv") or args.dataset.endswith(
        ".csv"
    ), "Dataset file must be a tsv or csv file, containing a list of files to be augmented"
    assert args.output.endswith(".json"), "Output file must be a json file"

    return args


def read_data(args):
    # Use the `dtype` parameter of `pd.read_csv`.
    files = pd.read_csv(args.dataset, sep=",", header=None, names=["file"])
    files = files["file"].tolist()
    data = []
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            data.append(text)

    return data


def get_augmenter(args):
    if args.augmenter == "essayinstruction":
        augmenter = EssayInstructor()

    elif args.augmenter == "essayrevision":
        augmenter = EssayReviser()

    elif args.augmenter == "stackexchange":
        augmenter = StackExchangeBuilder()

    elif args.augmenter == "hierarchicalsummarizer":
        augmenter = HierachicalSummarizer()

    elif args.augmenter == "entityrecognizedsummarizer":
        augmenter = EntityRecognizedSummarizer()

    elif args.augmenter == "codebugger":
        augmenter = CodeBugger()

    elif args.augmenter == "codeinstructor":
        augmenter = CodeInstructor()

    else:
        raise ValueError(
            "Augmenter must be one of 'essayinstruction', 'essayrevision', 'stackexchange', 'hierarchicalsummarizer', 'entityrecognizedsummarizer', 'codebugger', 'codeinstructor"
        )

    return augmenter


def main(args):
    data = read_data(args)
    augmenter = get_augmenter(args)

    augmented_data = augmenter.parse(data)

    # write augmented data as json file
    with open(args.output, "w") as f:
        json.dump(augmented_data, f)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
