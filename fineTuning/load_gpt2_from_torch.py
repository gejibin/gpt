import tiktoken
import torch
from gpt2.gpt2  import GPTModel, GPT_CONFIG_124M
from tools.tools import generate, text_to_token_ids, token_ids_to_text
from tools.download import download_file

def load_gpt2_from_torch(file_name):
    # file_name = "gpt2-small-124M.pth"
    # file_name = "gpt2-medium-355M.pth"
    # file_name = "gpt2-large-774M.pth"
    # file_name = "gpt2-xl-1558M.pth"
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

    if not download_file(url, file_name):
        return None
    
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update({"qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG)
    gpt.load_state_dict(torch.load(file_name, weights_only=True))
    gpt.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt.to(device)
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate(
        model=gpt.to(device),
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    file_name = "gpt2-small-124M.pth"
    # file_name = "gpt2-medium-355M.pth"
    # file_name = "gpt2-large-774M.pth"
    # file_name = "gpt2-xl-1558M.pth"
    load_gpt2_from_torch(file_name)



