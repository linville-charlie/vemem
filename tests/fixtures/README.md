# Test fixtures — face images

This directory holds small public-domain portraits used by the encoder
integration tests (see `tests/encoders/test_insightface.py`). Only historical
figures whose photos are unambiguously **public domain** are used, so the
library can be freely redistributed and CI can fetch the repo without a
licensing gate.

All files were downloaded from Wikimedia Commons at 480px thumbnail width.
Each image is described below with its provenance.

## Images

### `person_a_front.jpg`
- **Subject:** Albert Einstein — frontal head-and-shoulders portrait.
- **Source:** [Albert Einstein Head (Wikimedia Commons)](https://commons.wikimedia.org/wiki/File:Albert_Einstein_Head.jpg)
- **Direct URL:** `https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/480px-Albert_Einstein_Head.jpg`
- **License:** Public Domain (photograph first published before 1929).
- **Test role:** anchor for "same person" similarity tests — the frontal
  reference against which `person_a_alt.jpg` must match strongly.

### `person_a_alt.jpg`
- **Subject:** Albert Einstein — alternate portrait (colorized).
- **Source:** [Albert Einstein - Colorized (Wikimedia Commons)](https://commons.wikimedia.org/wiki/File:Albert_Einstein_-_Colorized.jpg)
- **Direct URL:** `https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Albert_Einstein_-_Colorized.jpg/480px-Albert_Einstein_-_Colorized.jpg`
- **License:** Public Domain (underlying photograph pre-1929; colorization
  is a derivative of a PD work).
- **Test role:** "same person, different pose/age" — embedding cosine with
  `person_a_front.jpg` must pass the ArcFace similarity threshold.

### `person_b_front.jpg`
- **Subject:** Nikola Tesla — frontal portrait, ca. 1890.
- **Source:** [Tesla circa 1890 (Wikimedia Commons)](https://commons.wikimedia.org/wiki/File:Tesla_circa_1890.jpeg)
- **Direct URL:** `https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Tesla_circa_1890.jpeg/480px-Tesla_circa_1890.jpeg`
- **License:** Public Domain (photograph first published before 1929).
- **Test role:** "different person" control — embedding cosine with any
  Einstein fixture must fall below the ArcFace threshold.

### `person_c_front.jpg`
- **Subject:** Marie Curie — portrait, ca. 1920s.
- **Source:** [Marie Curie c. 1920s (Wikimedia Commons)](https://commons.wikimedia.org/wiki/File:Marie_Curie_c._1920s.jpg)
- **Direct URL:** `https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Marie_Curie_c._1920s.jpg/480px-Marie_Curie_c._1920s.jpg`
- **License:** Public Domain (photograph first published before 1929).
- **Test role:** extra "different person" fixture — useful when adding
  cross-person similarity distribution checks.

## Adding more fixtures

Preferred sources:

- **Wikimedia Commons** — prefer images in the `Public domain` category
  (especially `PD-old-expired` / `PD-US-expired`) or explicit CC0.
- If you contribute a fixture of a living person (including yourself),
  add a `LICENSE-<name>.txt` in this directory explaining the consent and
  license under which it's included.

For every image added, extend this README with the same fields
(Subject / Source / Direct URL / License / Test role).

## Regeneration

A one-shot fetch script is **intentionally not committed** — the images are
in the repo so they're always available for CI without network. If you need
to re-download or add new fixtures, fetch at 480px from Wikimedia Commons
with a descriptive `User-Agent` header (Wikimedia blocks the default
Python `urllib` UA). Commit the bytes to this directory and update this
README.
