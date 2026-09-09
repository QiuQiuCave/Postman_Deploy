# Slim cola-can visual asset

Generated using the built-in `imagegen` tool on 2026-09-09 for the user's requested Coca-Cola-style simulation prop. This is generated illustrative artwork, not an official packaging asset or an affiliation claim. Coca-Cola branding belongs to its respective owner.

## Files and physical scope

- `label.png`: final opaque 1254 × 1254 can-wrap image, copied unchanged from the generation result.
- `common/r2v2_can_visual.py`: repository-root-relative path to the UV shell, silver lids/rims and geometric pull-tab.
- The shell fits the original simulated cylinder: diameter 40 mm, height 120 mm, mass 100 g. These are the existing task dimensions, not a claim about a real commercial can's dimensions or volume.
- All appended visual geoms explicitly use `mass=0`, `density=0`, `contype=0`, `conaffinity=0`; original cylinder dynamics and contacts remain intact.
- The image wraps once around the cylinder. Logo centres at u=.25/.75 face cylinder-local -Y/+Y; v=0/1 is top/bottom, matching image loading. The silver ends and tab are native geometry, not baked into this label.
- Set `object_appearance: cola_can` in `deploy_mujoco/config/r2v2_tabletop_demo.yaml`; `orange_cylinder` restores the original appearance.

## Generation provenance

Mode: built-in image generation, then built-in image edit to repair unintended transparency. No image post-processing was used. The first draft is not shipped. The prompts below are the complete submitted prompts, including the final edit prompt.

### Initial generation prompt

```text
Use case: product-mockup
Asset type: flat unwrapped cylindrical UV label texture for a 3D slim Coca-Cola beverage can in a robotics simulation.
Primary request: create one square full-bleed texture image, approximately 1024 by 1024, of the recognizable classic red Coca-Cola can paint design.
Composition: flat 2D print artwork, NOT a photo of a can. Solid rich Coca-Cola red background edge to edge with no paper, mockup, border, metal rims or scene. The image maps once around the whole circumference and bottom-to-top up the can. Include TWO matching white classic flowing cursive Coca-Cola wordmarks, one centered in the left half and one in the right half. Each entire wordmark is rotated 90 degrees counterclockwise so it reads from bottom to top along the tall can; the lettering must remain authentic-looking cursive and spell exactly "Coca-Cola". Each rotated wordmark is about 75 percent of the canvas height and less than 30 percent of its width, with ample red breathing room at top and bottom. Small white text "ORIGINAL TASTE" near the top of each half and "可口可乐" near the bottom of each half, arranged cleanly. A restrained thin white wave ribbon may run along the bottom.
Lighting: uniform diffuse albedo only, absolutely no baked highlights, shadows, wrinkles or perspective.
Constraints: all logos completely inside canvas, red at left/right seam, crisp readable typography. No can silhouette, no silver top or bottom, no pull tab, no watermark, no extraneous labels, no nutrition panel, no barcode. This is a texture asset, not a rendered product picture.
```

### Final edit prompt

Input: the first generated label image.

```text
Edit this flat Coca-Cola can-wrap artwork to fix its broken transparent background. The output MUST be a completely OPAQUE RGB-style full rectangular red label, with no transparent pixels anywhere. Replace ALL transparent, black, speckled and distressed background regions with one flat vivid classic cola red (#E41E26). Preserve two large white Coca-Cola cursive wordmarks reading bottom-to-top, one centered in the left half and one in the right half. Retain the small white ORIGINAL TASTE and 可口可乐 lettering and the restrained white bottom wave. Make all white lettering clean, crisp and solid, with no outlines or drop shadows. This is a clean printed full-bleed 2D UV texture image, NOT a cutout, not a transparent logo, and not a photograph or rendered can. The ENTIRE square including all four corners, edges, and space between the logos must be filled with fully opaque flat red. Do not apply background removal. Do not output an alpha channel or transparent background.
```
