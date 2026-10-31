"""Keep the label-policy RNG unchanged when record coverage is enabled."""


def choose_training_record(rng, pool, label, *, coverage=None, record_by_id=None):
    legacy = rng.choice(pool)
    if coverage is None:
        return legacy
    selected = record_by_id[coverage.draw(label)]
    if selected["split"] != "train" or selected["label"] != label:
        raise ValueError("Coverage helper returned a non-training/wrong-class record")
    return selected
