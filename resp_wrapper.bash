#!/bin/bash
# resp_wrapper.bash
# hrmd, 01/03/2017.
# A wrapper script for the respknt script, part of Charles Ammon's
#   receiver function package. This wrapper is used by the earth_response() method of the LayeredModel class to feed arguments to the respknt command.

# Read input from command line.
# The first input, respknt, is the path to the respknt executable.
# The other inputs are explained below.
respknt=$1
model_file=$2
p_or_s=$3
delta=$4
length=$5
slowness=$6

# Feed commands to respknt using a Bash 'here function'.
# Inputs are:
# 1. Model file.
# 2. Print model file to stdout? (y or n.)
# 3. P- or S- response? (1 or 2.)
# 4. Sampling interval.
# 5. Duration of synthetic response.
# 6. Slowness of incident ray.
# 7. Full or partial calculation? (f or p.)
# 8. Mode conversions? (y or n)
${respknt} << EOF
${model_file}
n
${p_or_s}
${delta}
${length}
${slowness}
f
y

EOF
