import json
from typing import Tuple, Dict, Any

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import argparse
import pathlib
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()


def load_model() -> Tuple[Llama, Dict[str, Any]]:
    # Your existing model setup
    model_name = "unsloth/gemma-3-27b-it-GGUF"
    model_file = "gemma-3-27b-it-Q4_K_M.gguf"
    model_path = hf_hub_download(model_name, filename=model_file)

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=32,
        n_gpu_layers=999,
        seed=42,
    )

    generation_kwargs = {
        "max_tokens": 2000,
        "stop": ["</s>"],
        "echo": False,
        "top_k": 1,
        "temperature": 0.1
    }

    return llm, generation_kwargs


def count_lines(file_path: pathlib.Path) -> int:
    """Count non-empty lines in the JSONL file."""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def display_input_output(custom_id: str, messages: list, generated_text: str, line_num: int, total_lines: int):
    """Display current input and output in a formatted way."""

    # Create input display
    input_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # Truncate long content for display
        if len(content) > 200:
            content = content[:200] + "..."
        input_text += f"[bold cyan]{role}:[/bold cyan] {content}\n"

    # Truncate output for display
    output_display = generated_text
    if len(output_display) > 300:
        output_display = output_display[:300] + "..."

    # Create panels
    console.print(f"\n[bold green]Processing [{line_num}/{total_lines}]: {custom_id}[/bold green]")

    console.print(Panel(
        input_text.rstrip(),
        title="[bold blue]Input Messages[/bold blue]",
        border_style="blue",
        padding=(0, 1)
    ))

    console.print(Panel(
        output_display,
        title="[bold magenta]Generated Output[/bold magenta]",
        border_style="magenta",
        padding=(0, 1)
    ))


def process_jsonl_file(file_path: pathlib.Path, intermediate_path: pathlib.Path, llm: Llama, generation_kwargs: Dict[str, Any]) -> list:
    """Process a JSONL file with Korean news analysis tasks using chat completion."""
    results = []

    # Count total lines for progress bar
    total_lines = count_lines(file_path)
    console.print(f"[bold green]Found {total_lines} entries to process[/bold green]")

    # Create progress bar
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
    ) as progress:

        task = progress.add_task("[green]Processing JSONL entries...", total=total_lines)

        count = {
            "total": 0,
            "success": 0,
            "errors": 0,
            "pred_same": {
                "true_same": 0,
                "false_same": 0,
            },
            "pred_diff": {
                "true_diff": 0,
                "false_diff": 0,
            }
        }

        with open(file_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    custom_id = entry.get("custom_id", f"line_{line_num}")

                    # Extract messages and parameters from the original format
                    messages = entry["body"]["messages"]
                    body = entry["body"]

                    # Extract parameters from the original request
                    temperature = body.get("temperature", 0.1)
                    max_tokens = body.get("max_tokens", 2000)
                    response_format = body.get("response_format")

                    # Prepare chat completion parameters
                    chat_params = {
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_k": 1,
                        "stop": ["</s>"]
                    }

                    # Add response format if specified
                    if response_format and response_format.get("type") == "json_object":
                        chat_params["response_format"] = response_format

                    # Use chat completion directly with the messages
                    response = llm.create_chat_completion(**chat_params)

                    generated_text = response["choices"][0]["message"]["content"].strip()

                    # Display input and output
                    display_input_output(custom_id, messages, generated_text, line_num, total_lines)

                    # Try to parse as JSON if response_format was json_object
                    if response_format and response_format.get("type") == "json_object":
                        try:
                            parsed_response = json.loads(generated_text)
                        except json.JSONDecodeError:
                            # If not valid JSON, wrap in a structure
                            parsed_response = {"raw_text": generated_text, "parse_error": True}
                    else:
                        # For non-JSON responses, just store the text
                        parsed_response = {"content": generated_text}

                    result = {
                        "custom_id": custom_id,
                        "request": messages,
                        "response": parsed_response,
                        "raw_output": generated_text,
                        "finish_reason": response["choices"][0].get("finish_reason"),
                        "usage": response.get("usage", {})
                    }

                    results.append(result)

                    count["total"] += 1
                    count["success"] += 1
                    if "same" in custom_id:
                        if parsed_response and (
                            type(parsed_response["답변"]) == str and parsed_response["답변"].strip().lower() == "true" or
                            type(parsed_response["답변"]) == bool and parsed_response["답변"] is True
                        ):
                            count["pred_same"]["true_same"] += 1
                        else:
                            count["pred_same"]["false_same"] += 1
                    elif "diff" in custom_id:
                        if parsed_response and (
                            type(parsed_response["답변"]) == str and parsed_response["답변"].strip().lower() == "false" or
                            type(parsed_response["답변"]) == bool and parsed_response["답변"] is False
                        ):
                            count["pred_diff"]["true_diff"] += 1
                        else:
                            count["pred_diff"]["false_diff"] += 1
                    else:
                        console.print(f"[bold yellow]⚠️ Unrecognized custom_id format: {custom_id}[/bold yellow]")

                    # Log count statistics table
                    console.print(Panel(
                        f"[bold blue]Count Statistics[/bold blue]\n"
                        f"Total: {count['total']}\n"
                        f"Success: {count['success']}\n"
                        f"Errors: {count['errors']}\n"
                        f"Pred Same True: {count['pred_same']['true_same']}\n"
                        f"Pred Same False: {count['pred_same']['false_same']}\n"
                        f"Pred Diff True: {count['pred_diff']['true_diff']}\n"
                        f"Pred Diff False: {count['pred_diff']['false_diff']}",
                        title="Statistics",
                        border_style="cyan"
                    ))

                    # Save intermediate results to file
                    with open(intermediate_path, 'at', encoding='utf-8') as intermediate_file:
                        intermediate_file.write(json.dumps(result, ensure_ascii=False) + '\n')

                    # Display the result in a panel


                    # Update progress
                    progress.update(task, advance=1)

                    console.print(f"[dim]✅ Completed {custom_id}[/dim]")
                    console.print("-" * 80)

                except Exception as e:
                    console.print(f"[bold red]❌ Error processing line {line_num}: {e}[/bold red]")
                    results.append({
                        "custom_id": f"line_{line_num}",
                        "error": str(e)
                    })
                    progress.update(task, advance=1)

    return results


def save_results(results, output_file):
    """Save results to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSONL file with Korean news analysis tasks.")
    parser.add_argument("--input-file", type=str, help="Path to the input JSONL file",
                        default="./dataset/batch/batch.jsonl")
    parser.add_argument("--intermediate-file", type=str, help="Path to save intermediate results",
                        default="./dataset/batch/intermediate_results.jsonl")
    parser.add_argument("--output-file", type=str, help="Path to save the output JSONL file",
                        default="./dataset/batch/output.jsonl")

    args = parser.parse_args()

    input_file = pathlib.Path(args.input_file).resolve()
    intermediate_file = pathlib.Path(args.intermediate_file).resolve()
    output_file = pathlib.Path(args.output_file).resolve()

    # Ensure intermediate file is empty or does not exist
    if intermediate_file.exists():
        intermediate_file.unlink()

    # Display startup info
    console.print(Panel(
        f"[bold blue]JSONL Processor Starting[/bold blue]\n"
        f"Input: {input_file}\n"
        f"Output: {output_file}",
        title="Configuration",
        border_style="green"
    ))

    console.print("[yellow]Loading model...[/yellow]")
    model, res_generation_kwargs = load_model()
    console.print("[green]✅ Model loaded successfully![/green]")

    console.print("\n[bold yellow]Starting JSONL processing...[/bold yellow]")
    results = process_jsonl_file(input_file, intermediate_file, model, res_generation_kwargs)

    # Final summary
    successful = sum(1 for r in results if "error" not in r)
    failed = len(results) - successful

    console.print(Panel(
        f"[bold green]Processing Complete![/bold green]\n"
        f"Total entries: {len(results)}\n"
        f"Successful: {successful}\n"
        f"Failed: {failed}\n"
        f"Results saved to: {output_file}",
        title="Summary",
        border_style="green"
    ))

    save_results(results, output_file)
    console.print(f"[bold green]🎉 All done! Results saved to {output_file}[/bold green]")