# HuggingFace Unauthenticated Warning

## Question

Seeing this warning — do the scripts have to access HF every time they run after the model has been downloaded?

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

## Response

No — the model is already cached locally after the first download. That warning is just HuggingFace nudging you to create an account for higher rate limits. It's harmless and the scripts work fine without a token.

To suppress the warning:

```bash
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

Or just ignore it — it doesn't affect functionality.
