# Publishing TRACE

The public destination is `Amir-Ivry/trace`. GitHub Pages serves the committed
`docs/` directory at https://amir-ivry.github.io/trace/.

The repository contains the companion code, exact audio inputs, saved responses,
scripted scoring examples, and unchanged submitted manuscript. It needs no live
backend, API key, model weights, or GPU to build or browse.

## Build

From the repository root, using Python 3.10 or newer:

```bash
python3 -B scripts/prepare_github_pages.py
```

This builds `site/`, creates the code download and preview ZIP in `dist/`, and
replaces generated `docs/` with the finished site, including the code download.
Edit `templates/`, `configs/`, or `site/assets/`, then rebuild. Do not edit `docs/`
directly. Committed updates to `docs/` are what GitHub Pages publishes.

## First publication

1. Check whether `Amir-Ivry/trace` already exists. If so, inspect its default
   branch and any Pages configuration before adding these files; preserve
   unrelated work and use an ordinary commit, never a force push.
2. If it does not exist, create a public repository named `trace`, owned by
   `Amir-Ivry`, and use `main` as its default branch. Do not add a generated
   README or select a blanket license during creation.
3. Push the companion source and generated `docs/` in one ordinary commit.
4. In Settings → Pages, select **Deploy from a branch**, branch **main**, folder
   **/docs**, and save. Use the actual default branch if the repository already
   has another one.
5. Wait for GitHub's Pages deployment to succeed, then check the published page,
   its three audio players, paper link, GitHub link, and code download.

This does not require changes to `Amir-Ivry/amir-ivry.github.io`. Do not change
that repository, a custom domain, DNS, or unrelated account settings. If an
existing `/trace/` page or custom-domain configuration conflicts with the target
URL, inspect it before deciding how to proceed.

## Access needed

The publishing session needs access to create the repository if it is absent,
write repository contents, and configure GitHub Pages. Repository-specific access
is sufficient after the repository exists. The account owner can alternatively
create the empty `trace` repository and enable Pages personally. Credentials must
be handled through GitHub sign-in or an authenticated tool, never committed.

## Evidence and licensing

Publishing does not change the three saved responses or their review status.
Human listening and residual-cue review are still pending and disclosed on the
page. No new inference is required. Audio credits and CC0 terms remain attached;
no new code or manuscript reuse license is granted by this preparation. See
`CREDITS_AND_LICENSES.txt`.
