# MXB tutorials and scripts

A collection of jupyter notebooks and scripts for analyzing and simulation
genomic data, focusing on the MXB and MXL genomes.

## Package requirements

We'll be using a few standard packages for scientific computing, plotting, and
simulation, as well as a few packages that are in development or that we'll be
using new upcoming features from. Mostly, I'll assume we are using `conda` and
will install most packages that way, though a few we'll clone directly and
build locally from the dev branch or `pip install` some alpha version.

Note that we're pulling in the `msprime` alpha release for version 1.0, which
will be released 

```
conda create -n mxb_tutorial python=3.8 -y
pip install -r requirements.txt
```

`moments` is available via conda, but we'll be using some new API, inference,
and plotting features that are in development and haven't been pulled into
a tagged release quite yet. This is why we install from the `devel` branch.
Risky! But I'm the main developer of `moments` and will make sure everything
stays working.

Similarly, `demesdraw` is a package for plotting nice representations of
demographic models using the `demes` specification, developed primarily
by Graham Gower. That will also get a proper release soon, but we'll also
pull from git for that package as well.
