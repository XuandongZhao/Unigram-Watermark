import argparse
from tqdm import tqdm
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LogitsProcessorList
from gptwm import GPTWatermarkLogitsWarper


def read_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]


def write_file(filename, data):
    with open(filename, "a") as f:
        f.write("\n".join(data) + "\n")


def main(args):
    output_file = f"{args.output_dir}/{args.model_name.replace('/', '-')}_strength_{args.strength}_frac_{args.fraction}_len_{args.max_new_tokens}_num_{args.num_test}.jsonl"
    if 'llama' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')
    model.eval()

    watermark_processor = LogitsProcessorList([GPTWatermarkLogitsWarper(fraction=args.fraction,
                                                                        strength=args.strength,
                                                                        vocab_size=model.config.vocab_size,
                                                                        watermark_key=args.wm_key)])

    data = read_file(args.prompt_file)
    num_cur_outputs = len(read_file(output_file)) if os.path.exists(output_file) else 0

    outputs = []

    for idx, cur_data in tqdm(enumerate(data), total=min(len(data), args.num_test)):
        if idx < num_cur_outputs or len(outputs) >= args.num_test:
            continue

        if "gold_completion" not in cur_data and 'targets' not in cur_data:
            continue
        elif "gold_completion" in cur_data:
            prefix = cur_data['prefix']
            gold_completion = cur_data['gold_completion']
        else:
            prefix = cur_data['prefix']
            gold_completion = cur_data['targets'][0]

        batch = tokenizer(prefix, truncation=True, return_tensors="pt")
        num_tokens = len(batch['input_ids'][0])

        with torch.inference_mode():
            generate_args = {
                **batch,
                'logits_processor': watermark_processor,
                'output_scores': True,
                'return_dict_in_generate': True,
                'max_new_tokens': args.max_new_tokens,
            }

            if args.beam_size is not None:
                generate_args['num_beams'] = args.beam_size
            else:
                generate_args['do_sample'] = True
                generate_args['top_k'] = args.top_k
                generate_args['top_p'] = args.top_p

            generation = model.generate(**generate_args)
            gen_text = tokenizer.batch_decode(generation['sequences'][:, num_tokens:], skip_special_tokens=True)

        outputs.append(json.dumps({
            "prefix": prefix,
            "gold_completion": gold_completion,
            "gen_completion": gen_text
        }))

        if (idx + 1) % 100 == 0:
            write_file(output_file, outputs)
            outputs = []

    write_file(output_file, outputs)
    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--model_name", type=str, default="decapoda-research/llama-7b-hf")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--prompt_file", type=str, default="./data/LFQA/inputs.jsonl")
    parser.add_argument("--output_dir", type=str, default="./data/LFQA/")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()
    main(args)
