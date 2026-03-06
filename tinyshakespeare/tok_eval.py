"""
Evaluate compression ratio of the tokenizer.
"""
import os
from nanochat.tokenizer import get_tokenizer, RustBPETokenizer
from nanochat.tinyshakespeare import load_dataset

from dotenv import load_dotenv
load_dotenv()
os.environ["NANOCHAT_BASE_DIR"] = os.getenv("TS")

# Random text I got from a random website this morning
news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico’s National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation’s food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

“The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening’s to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border,” said U.S. Secretary of Agriculture Brooke L. Rollins. “Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest.”
""".strip()

# Excerpt from Thomas More's "Utopia" (17th century)
utopia_text = r"""
OF THEIR TRADES, AND MANNER OF LIFE

Agriculture is that which is so universally understood among them that no person, either man or woman, is ignorant of it; they are instructed in it from their childhood, partly by what they learn at school, and partly by practice, they being led out often into the fields about the town, where they not only see others at work but are likewise exercised in it themselves. Besides agriculture, which is so common to them all, every man has some peculiar trade to which he applies himself; such as the manufacture of wool or flax, masonry, smith’s work, or carpenter’s work; for there is no sort of trade that is in great esteem among them. Throughout the island they wear the same sort of clothes, without any other distinction except what is necessary to distinguish the two sexes and the married and unmarried. The fashion never alters, and as it is neither disagreeable nor uneasy, so it is suited to the climate, and calculated both for their summers and winters. Every family makes their own clothes; but all among them, women as well as men, learn one or other of the trades formerly mentioned. Women, for the most part, deal in wool and flax, which suit best with their weakness, leaving the ruder trades to the men. The same trade generally passes down from father to son, inclinations often following descent: but if any man’s genius lies another way he is, by adoption, translated into a family that deals in the trade to which he is inclined; and when that is to be done, care is taken, not only by his father, but by the magistrate, that he may be put to a discreet and good man: and if, after a person has learned one trade, he desires to acquire another, that is also allowed, and is managed in the same manner as the former. When he has learned both, he follows that which he likes best, unless the public has more occasion for the other.
""".strip()

# Excerpt from Oscar Wilde's "The Importance of Being Honest" (19th century)
wilde_text = r"""
CECILY.
Uncle Jack, you are not going to refuse your own brother’s hand?

JACK.
Nothing will induce me to take his hand. I think his coming down here disgraceful. He knows perfectly well why.

CECILY.
Uncle Jack, do be nice. There is some good in every one. Ernest has just been telling me about his poor invalid friend Mr. Bunbury whom he goes to visit so often. And surely there must be much good in one who is kind to an invalid, and leaves the pleasures of London to sit by a bed of pain.

JACK.
Oh! he has been talking about Bunbury, has he?

CECILY.
Yes, he has told me all about poor Mr. Bunbury, and his terrible state of health.

JACK.
Bunbury! Well, I won’t have him talk to you about Bunbury or about anything else. It is enough to drive one perfectly frantic.

ALGERNON.
Of course I admit that the faults were all on my side. But I must say that I think that Brother John’s coldness to me is peculiarly painful. I expected a more enthusiastic welcome, especially considering it is the first time I have come here.

CECILY.
Uncle Jack, if you don’t shake hands with Ernest I will never forgive you.

JACK.
Never forgive me?

CECILY.
Never, never, never!

JACK.
Well, this is the last time I shall ever do it.
""".strip()

# The tokenizer was trained on data from earlier shards, so it has seen this data
# train_docs = next(parquets_iter_batched(split="train"))
# train_text = "\n".join(train_docs)
# val_docs = next(parquets_iter_batched(split="val"))
# val_text = "\n".join(val_docs)
train_text, val_text = load_dataset() 

all_text = [
    ("news", news_text),
    ("utopia", utopia_text),
    ("wilde", wilde_text),
    ("fwe-train", train_text),
]
if val_text:
    all_text.append(("fwe-val", val_text))

# Try out current default compared to GPT-2 and GPT-4 tokenizers
tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt2", "gpt4", "ours"]:

    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2") # gpt-2 base model tokenizer
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base") # gpt-4 base model tokenizer
    else:
        tokenizer = get_tokenizer()

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

        encoded_bytes = text.encode('utf-8')
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'ratio': ratio
        }

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# Print vocab sizes
print(f"\nVocab sizes:")
print(f"GPT-2: {vocab_sizes['gpt2']}")
print(f"GPT-4: {vocab_sizes['gpt4']}")
print(f"Ours: {vocab_sizes['ours']}")

def print_comparison(baseline_name, baseline_results, ours_results, all_text):
    """Print comparison table between baseline tokenizer and ours."""
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        # Calculate relative difference (positive means ours is better, negative means worse)
        # Using tokens: fewer tokens is better, so we calculate (baseline_tokens - ours_tokens) / baseline_tokens
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100

        # Determine which has better compression (higher ratio = better)
        if baseline_data['ratio'] > ours_data['ratio']:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data['ratio'] > baseline_data['ratio']:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(f"{name:<10} {baseline_data['bytes']:<8} "
              f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
              f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
              f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
              f"{better:<10}")

# Print comparisons
print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)

# Log to report
from nanochat.report import get_report
lines = []
for baseline_name in ["GPT-2", "GPT-4"]:
    baseline_key = baseline_name.lower().replace('-', '')
    baseline_results = tokenizer_results[baseline_key]
    ours_results = tokenizer_results['ours']
    lines.append(f"### Comparison with {baseline_name}")
    lines.append("")
    lines.append("| Text Type | Bytes | " + baseline_name + " Tokens | " + baseline_name + " Ratio | Ours Tokens | Ours Ratio | Relative Diff % |")
    lines.append("|-----------|-------|--------------|--------------|-------------|------------|-----------------|")
    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
        lines.append(f"| {name} | {baseline_data['bytes']} | {baseline_data['tokens']} | {baseline_data['ratio']:.2f} | {ours_data['tokens']} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |")
    lines.append("")
report_markdown = "\n".join(lines)
get_report().log(section="Tokenizer evaluation", data=[
    report_markdown,
])
