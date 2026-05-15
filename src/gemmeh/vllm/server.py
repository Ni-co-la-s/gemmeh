"""
Server to run trained model model on vllm.
safetensor model file and tokenizer files needs to be in the provided folder
Example usage:
uv run -m gemmeh.vllm.server --model /path/to/models
Optional arguments port, served_model_name, gpu_memory_utilization, dtype and
max_num_seqs can also be provided
"""

import asyncio


from vllm.entrypoints.openai.api_server import run_server, make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
import gemmeh.vlm.model  # noqa: F401


def main():
    parser = make_arg_parser(FlexibleArgumentParser())

    parser.set_defaults(
        port=8000,
        served_model_name=["gemmeh"],
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        max_num_seqs=32,
        # --enable-tokenizer-info-endpoint would be necessary for lm-eval
    )

    args = parser.parse_args()
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
