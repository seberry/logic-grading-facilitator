"""Common proof OCR building blocks.

Recursive decomposition workflow:
1. Detect vertical scope bars in the cropped proof image.
2. Build a subproof tree from those bars.
3. Recurse into child scopes first.
4. Mask each solved child with a synthetic placeholder token.
5. OCR the remaining visible rows for the current scope.
6. Stitch placeholders back into a nested proof structure.

This package is meant for mechanics shared across Fitch-style proof OCR,
independent of whether the target domain is TFL, modal, or another variant.
"""

