# Wasi-NN for Edge ML Inference: Demo Library

*Bachmeier, Yussupov, Henß, Koziolek: "Wasi-NN for Edge Machine
Learning Inference: Experiences and Lessons Learned". 2025. Accepted
for publication.*

## Authors

- Joshua Bachmeier [@joshuabach](https://github.com/joshuabach)
- Vladimir Yussupov [@v-yussupov](https://github.com/v-yussupov)
- Jörg Henß [@joehen](https://github.com/joehen)
- Heiko Koziolek [@hkoziolek](https://github.com/hkoziolek)

---

The proof-of-concept demo consists of two parts: First, a library
providing utilities for running time series models as a HTTP service
in WASM. Second, the actual
[component](https://github.com/joshuabach/wasi-nn-edge-demo) that uses
the library to run a concrete forecast on data collected by an edge
device. In this scenario, the library is provided by the industrial
automation provider and the component by the customer.

This is the prototype for the **library**. The best source of
information and usage instructions is the documentation, which can be
generated using `cargo` (see below under [Usage](#Usage)).

## Prerequisites
This library requires Rust nightly due to the usage of the unstable
feature
[`slice_as_chunks`](https://github.com/rust-lang/rust/issues/74985).
If required, usage of this feature could be removed without
compromising functionality of this library.

Make sure you have a working [Rust nightly
installation](https://www.rust-lang.org/tools/install), either by
using a distribution package, or with `rustup` (recommended). If you
use rustup, any `cargo` command should automatically install the
required components from the correct rust release channel due to the
definitions in [rust-toolchain.toml](rust-toolchain.toml) in this
repository. If necessary, the usage of these features could be removed
without compromising functionality.


## Usage

Include this library in your projects `Cargo.toml` like this (you may
need to adapt the path to this repository):

```toml
[dependencies]
wasi-nn-demo-lib = { path = "../wasi-nn-demo-lib" }
```

This library is intended to be used to build [WASM
components](https://component-model.bytecodealliance.org/) only, so
compile your project with:

```bash
cargo build --target=wasm32-wasip2
```

**Do not** use
[cargo-component](https://github.com/bytecodealliance/cargo-component),
which is a legacy approach to build WASM components that was created
before the `wasm32-wasip2` target was stabilized. Today, it is
unnecessary.

For an example component, see the repository [joshuabach/wasi-nn-edge-demo](https://github.com/joshuabach/wasi-nn-edge-demo).

Documentation can be generated and viewed with:

```bash
cargo doc --open
```

To verify that the examples included throughout the documentation are
actually working, use:

```bash
cargo test
```
