import argparse
import json
import random
import re
import sys
from uuid import uuid4

import pydantic
from oasst_data import ExportMessageNode, ExportMessageTree
from sampling_report import SamplingReport


def filter_text(s: str) -> str:
    m = re.search(
        r"\</?prefix\>|\<human\>|\<\|endoftext\|\>|\<\|prompter\|\>|\<\|assistant\|\>|\<\|system\|\>|<|prefix_(begin|end)\|\>",
        s,
    )
    if m:
        s = s[: m.start()]
    return s


def format_params(p: dict) -> str:
    s = [f"{k}={v}" for k, v in p.items()]
    return ",".join(s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="*", type=argparse.FileType("r", encoding="UTF-8"))
    parser.add_argument("--seed", default=219837, type=int)
    parser.add_argument("--num-replies", default=5, type=int)
    parser.add_argument("--output", type=argparse.FileType("w", encoding="UTF-8"), default=sys.stdout)
    parser.add_argument("--max-count", type=int)
    parser.add_argument("--lang", type=str, default="en")
    return parser.parse_args()


def main():
    args = parse_args()

    input_reports: list[SamplingReport] = []
    for f in args.input_files:
        json_raw = json.load(f)
        report = pydantic.parse_obj_as(SamplingReport, json_raw)
        input_reports.append(report)

    print(f"Read {len(input_reports)} reports")

    # index by prompt
    reply_by_prompt: dict[str, list[ExportMessageNode]] = {}
    for r in input_reports:
        for p in r.prompts:
            for res in p.results:
                for s in res.outputs:
                    s = filter_text(s)
                    model_name = f"{r.model_name},{format_params(res.sampling_params)}"
                    m = ExportMessageNode(
                        message_id=str(uuid4()),
                        text=s,
                        role="assistant",
                        synthetic=True,
                        model_name=model_name,
                        lang=args.lang,
                    )

                    l = reply_by_prompt.get(p.prompt)
                    if l is not None:
                        l.append(m)
                    else:
                        reply_by_prompt[p.prompt] = [m]

    random.seed(args.seed)
    trees: list[ExportMessageTree] = []
    for k, v in reply_by_prompt.items():
        # remove exact duplicates
        reply_texts = set()
        unique_replies = []
        for m in v:
            if m.text in reply_texts:
                continue
            unique_replies.append(m)
            reply_texts.add(m.text)

        if len(unique_replies) < 2:
            print("Skipping entry with < 2 unique replies")
            continue

        prompt_message = ExportMessageNode(
            message_id=str(uuid4()), text=k, role="prompter", synthetic=False, lang=args.lang
        )
        prompt_message.replies = random.sample(unique_replies, k=min(args.num_replies, len(unique_replies)))
        t = ExportMessageTree(message_tree_id=prompt_message.message_id, tree_state="ranking", prompt=prompt_message)
        trees.append(t)
        if args.max_count and len(trees) >= args.max_count:
            break

    with args.output as f:
        for t in trees:
            json.dump(t.dict(exclude_none=True), f)
            f.write("\n")


if __name__ == "__main__":
    main()
