# Concat Ascend C Operator

This directory contains the Ascend C implementation of basic FP16 concat.
The kernel accepts a device-resident input-address table and length table and
copies contiguous input segments into the output. The basic version supports
concatenation along the last dimension; the host shape helper validates that
constraint and matching non-concat dimensions.

Build it in an Ascend CANN environment:

```bash
export ASCENDC_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
./build.sh
```

`build.sh` does not create placeholder artifacts. After the Ascend build
system produces a real `custom_*.run`, the script copies it to
`build_out/custom_*.run`, which is the path consumed by `zip_op.sh`.
