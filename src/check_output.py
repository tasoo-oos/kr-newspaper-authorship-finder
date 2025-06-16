import json
from typing import Tuple, Dict, Any
import argparse
import pathlib


def check_output(file_path: str):
    ids_set = set()

    count = {
        "total": 0,
        "success": 0,
        "errors": 0,
        "pred_same": {
            "true_same": 0,
            "true_diff": 0,
        },
        "pred_diff": {
            "true_same": 0,
            "true_diff": 0,
        }
    }

    with open(file_path, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            custom_id = entry["custom_id"]

            if custom_id in ids_set:
                print(f"Duplicate ID found: {custom_id} at line {line_num}")

            ids_set.add(custom_id)

            count["total"] += 1

            if "parse_error" in entry["response"].keys():
                count["errors"] += 1
                print(f"Error found for ID {custom_id} at line {line_num}")

            else:
                count["success"] += 1

                pred_label = entry["response"].get("답변", None)


                if pred_label:
                    if "same" in custom_id:
                        count["pred_same"]["true_same"] += 1
                    else:
                        count["pred_same"]["true_diff"] += 1

                elif not pred_label:
                    if "same" in custom_id:
                        count["pred_diff"]["true_same"] += 1
                    else:
                        count["pred_diff"]["true_diff"] += 1

    print(f"Total unique IDs found: {len(ids_set)}")

    print(f"Total entries processed: {count['total']}")
    print(f"Successful entries: {count['success']}")
    print(f"Entries with errors: {count['errors']}")

    # confusion matrix
    print("Confusion Matrix:")
    print("-" * 40)
    print(f"Predicted Same (True Same): {count['pred_same']['true_same']}")
    print(f"Predicted Same (True Diff): {count['pred_same']['true_diff']}")
    print(f"Predicted Diff (True Same): {count['pred_diff']['true_same']}")
    print(f"Predicted Diff (True Diff): {count['pred_diff']['true_diff']}")
    print("-" * 40)

    # Get Accuracy, Precision, Recall, F1 Score, MCC
    total_pred_same = count['pred_same']['true_same'] + count['pred_same']['true_diff']
    total_pred_diff = count['pred_diff']['true_same'] + count['pred_diff']['true_diff']
    accuracy = (count['pred_same']['true_same'] + count['pred_diff']['true_diff']) / count['total']
    precision = (count['pred_same']['true_same']) / total_pred_same if total_pred_same > 0 else 0
    recall = (count['pred_same']['true_same']) / (count['pred_same']['true_same'] + count['pred_diff']['true_same']) if (count['pred_same']['true_same'] + count['pred_diff']['true_same']) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    mcc_numerator = (count['pred_same']['true_same'] * count['pred_diff']['true_diff']) - (count['pred_same']['true_diff'] * count['pred_diff']['true_same'])
    mcc_denominator = ((count['pred_same']['true_same'] + count['pred_same']['true_diff']) *
                       (count['pred_diff']['true_same'] + count['pred_diff']['true_diff']) *
                      (count['pred_same']['true_same'] + count['pred_diff']['true_same']) *
                      (count['pred_same']['true_diff'] + count['pred_diff']['true_diff'])) ** 0.5

    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")
    print(f"MCC:       {mcc:.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check JSONL output for duplicates and missing fields.")
    parser.add_argument("--file-path", type=str, help="Path to the JSON output file.", default="./dataset/batch/output.jsonl")
    args = parser.parse_args()

    file_path = pathlib.Path(args.file_path)
    if not file_path.is_file():
        print(f"File not found: {file_path}")
    else:
        check_output(file_path)