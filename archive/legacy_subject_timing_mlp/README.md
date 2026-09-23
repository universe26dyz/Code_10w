# Archived legacy subject/timing-pool MLP workflow

This directory preserves retired source, configurations, run scripts, tests and
documentation without deleting history.  It is deliberately outside the active
Python package and archived tests use a `.py.disabled` suffix so they are not
collected by pytest.

The archived workflow trained the surrogate from real-subject timing pools or
fixed `CYJ/DYZ -> HHZ -> HJL` subject splits.  It is not compatible with the
current experiment: offline training uses only a fixed HHZ/VPS=87,
rhythm-disjoint RR synthetic dataset.  Online reconstruction is per subject,
uses all valid SAX/2CH/4CH observations, and invokes the shared reconstruction
engine through either the Bloch or FrozenMLP decoder factory.

Do not import or run files in this directory for a current experiment.
