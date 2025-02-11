#!/bin/sh
# Run the example script in unbuffered mode, sending its output (logs) to stderr
python -u example_mnist.py 1>&2

# Create a zip archive of the output folder
zip -r results.zip output

# Output the zip archive to stdout (so that it can be captured)
cat results.zip
