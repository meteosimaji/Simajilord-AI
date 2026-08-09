# Meteo generated datasets

This directory is intentionally empty in Git. Teacher replays, calibrated mixtures, game
trajectories, checkpoints, and sealed evaluation positions are generated locally and can be
hundreds of megabytes. They are not source code, and the sealed final-test must not be exposed to
training or published with the repository.

Every publishable run records source URLs, rights profile IDs, engine and artifact SHA-256 values,
input replay hashes, normalized-position split rules, search settings, calibration formulae, parent
checkpoint hashes, and code revision in a small manifest. A dataset or champion checkpoint may be
uploaded as a separate release artifact only after that manifest, its redistribution boundary, and
the held-out evaluation gate have been reviewed.

Local generated files below this directory are ignored by Git. Do not force-add them merely to
make a run appear reproducible; publish the recreating command and hash chain, and use an explicit
artifact store for data that is both rights-clean and intended for distribution.
