# genmo/mochi-1-lora Cog Model

This is an implementation of [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview) as a [Cog](https://github.com/replicate/cog) model for LoRA inference.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).


## How to use

Make sure you have [cog](https://github.com/replicate/cog) installed.

To run a prediction:

    cog predict -i prompt="The video opens with a close-up of a woman in a white and purple outfit, holding a glowing purple butterfly. She has dark hair and walks gracefully through a traditional Japanese-style village at night" -i hf_lora="svjack/mochi_game_mix_early_lora"

![Output](output.gif)