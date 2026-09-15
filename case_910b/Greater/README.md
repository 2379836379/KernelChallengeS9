# Greater Ascend C Operator

This directory contains an Ascend C implementation of elementwise `x1 > x2`
for FP16 inputs. The device kernel supports flattened tensors and writes one
byte per boolean result. Broadcasting and shape inference are handled by the
host metadata helper.

## Build

The CANN toolkit and Ascend C compiler must be installed first:

```bash
export ASCENDC_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
./build.sh
```

The current container does not provide that toolchain, so compilation must be
performed in an Ascend development environment. `build.sh` requires the real
build to produce `custom_*.run`, then copies that artifact to `build_out/`. It
never creates a placeholder package. `zip_op.sh` packages the `op_host/`,
`op_kernel/` and generated `build_out/custom_*.run` artifacts.
