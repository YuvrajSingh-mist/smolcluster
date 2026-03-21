import re
from typing import List, Tuple

import torch
from datasets import load_dataset


class QAPairsDataset(torch.utils.data.Dataset[Tuple[str, str]]):
    def __init__(self, questions: List[str], answers: List[str]):
        self.questions = questions
        self.answers = answers

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.questions[index], self.answers[index]


def extract_answer_from_gsm8k(gsm8k_answer: str) -> str:
    match = re.search(r"### (.*)", gsm8k_answer)
    if match:
        return match.group(1).strip()
    raise ValueError(f"Could not extract final answer from GSM8K answer: {gsm8k_answer}")


def build_train_val_dataloaders(
    batch_size: int,
    data_config: dict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset = load_dataset(
        data_config["dataset_name"],
        data_config.get("subset"),
    )
    train_split = dataset[data_config["train_split"]]
    val_split = dataset[data_config["val_split"]]

    train_questions: List[str] = list(train_split["question"])
    train_answers: List[str] = [extract_answer_from_gsm8k(ans) for ans in train_split["answer"]]
    val_questions: List[str] = list(val_split["question"])
    val_answers: List[str] = [extract_answer_from_gsm8k(ans) for ans in val_split["answer"]]

    train_dataset = QAPairsDataset(train_questions, train_answers)
    val_dataset = QAPairsDataset(val_questions, val_answers)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_dataloader, val_dataloader