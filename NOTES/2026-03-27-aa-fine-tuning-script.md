# Fine-Tuning Script for Custom Dog Words

## Question

I created the file data/fine-tuning.txt which has two columns in CSV format — word or phrase and reaction by the dog. Can you write something that will apply fine-tuning with the contents of this file? It should not overwrite the original GPT-2 file. How does the new result get saved?

## Data File Format

```
word or phrase,reaction
dinner,high excitement and walks into kitchen where food is usually prepared
walk,high excitement and follows owner to get things needed to leave for the walk
outside,high excitement and tries expects to be let outside
this way,quick attention to owner and knows often the next action is to walk a different direction...
frisbee,excitement to play and the dog looks around for the frisbee in the area
to go,excitement to go along with the person speaking somewhere
look,high attentiveness to the person speaking and where they may be pointing
```

## How Saving Works

The fine-tuned model is saved to `models/stormi/` using `model.save_pretrained()`. The original GPT-2 weights in `~/.cache/huggingface/` are completely untouched. To use the fine-tuned model, load from the local folder instead of `"gpt2"`:

```python
# Original:
model = GPT2LMHeadModel.from_pretrained("gpt2")
# Fine-tuned:
model = GPT2LMHeadModel.from_pretrained("models/stormi")
```

## Script: 18-fine_tune.py

Generates 5 varied sentence templates per word/reaction pair (35 training examples total), trains for 5 epochs using standard next-token prediction loss.

Template examples for a word/reaction pair:
```
When the owner said '{word}', the dog knew it was time to {reaction}.
The dog heard its owner say '{word}' and immediately felt ready to {reaction}.
The dog recognized '{word}', which meant {reaction}.
Hearing '{word}', the dog's reaction was to {reaction}.
The word '{word}' always made the dog {reaction}.
```

## Next Steps After Training

Run `06-trigger_word.py` and `09-word_cluster.py` pointed at `models/stormi` to compare before/after — "frisbee", "look", "this way" should behave differently in the fine-tuned model. The interpretability tools become a before/after comparison showing how the model learned the new associations.
