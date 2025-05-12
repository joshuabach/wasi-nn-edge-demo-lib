//! High-level WASI-NN wrappers
//!
//! Contains high-level wrappers around low level bindings imported
//! from the [`wasi_nn`] crate. After 0.6.0, the `wasi-nn` crate
//! switched from witx to wit for generating bindings (which is
//! required for wasip2). However, the bindings exported by that crate
//! are now 1:1 mappings of the low-level wit-API.
//!
//! So we provide some definitions to help with that, mostly using the
//! newtype pattern.
//!
//! As a starting point in this module, use [`GraphBuilder`] to
//! construct a model that you can execute.
use std::collections::HashMap;
use std::{fmt, mem};
use std::{io, path::Path};

use wasi_nn::{errors, graph, inference, tensor};

pub use graph::{ExecutionTarget, GraphEncoding};

/// Build a Graph based on serveral options using a builder pattern
///
/// Start with [`GraphBuilder::default`] and work your way from there, in the end call [`GraphBuilder::build`].
///
/// For example:
/// ```no_run
/// # use wasi_nn_demo_lib::nn::*;
/// let graph = GraphBuilder::default()
///     .encoding(GraphEncoding::Openvino)
///     .execution_target(ExecutionTarget::Cpu)
///     .from_files(["model_path", "weights_path"])?
///     .build()?;
/// # Ok::<(), Error>(())
/// ```
pub struct GraphBuilder {
    encoding: GraphEncoding,
    execution_target: graph::ExecutionTarget,
    model_data: Vec<Vec<u8>>,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self {
            encoding: GraphEncoding::Openvino,
            execution_target: graph::ExecutionTarget::Cpu,
            model_data: vec![],
        }
    }
}

impl GraphBuilder {
    /// Set the graph encoding (optional)
    ///
    /// By default, this is [`GraphEncoding::Openvino`]
    pub fn encoding(mut self, encoding: GraphEncoding) -> GraphBuilder {
        self.encoding = encoding;
        self
    }

    /// Set the execution target (optional)
    ///
    /// By default, this is [`ExecutionTarget::Cpu`]
    pub fn execution_target(mut self, execution_target: graph::ExecutionTarget) -> GraphBuilder {
        self.execution_target = execution_target;
        self
    }

    /// Load model data from files (mandatory)
    ///
    /// Loads model data from a sequence of files, and sets it for the graph.
    ///
    /// # Errors
    /// Returns [`Error::IOError`] in case of an I/O error reading the files
    pub fn from_files(
        mut self,
        model_files: impl IntoIterator<Item = impl AsRef<Path>>,
    ) -> Result<GraphBuilder, Error> {
        self.model_data = model_files
            .into_iter()
            .map(std::fs::read)
            .collect::<Result<_, _>>()?;
        Ok(self)
    }

    /// Finally build the graph
    pub fn build(self) -> Result<Graph, Error> {
        graph::load(&self.model_data, self.encoding, self.execution_target)
            .map(Graph)
            .map_err(From::from)
    }
}

/// A neural network model / graph for performing inference
///
/// To actually execute the model, get an [`ExecutionContext`] with
/// [`Graph::init_execution_context`]
pub struct Graph(graph::Graph);

impl Graph {
    /// Create an execution context for a run of the model.
    pub fn init_execution_context(self) -> Result<ExecutionContext, Error> {
        let Graph(graph) = self;
        graph
            .init_execution_context()
            .map(ExecutionContext)
            .map_err(From::from)
    }
}

/// An execution context for a graph
///
/// Binds a set of input tensors to a set of inferred output tensors.
pub struct ExecutionContext(inference::GraphExecutionContext);

impl ExecutionContext {
    /// Set the the input tensor with `name` to the given tensor.
    ///
    /// The names of the input tensors depend on your model and should
    /// be found in its documentation.
    pub fn set_input<T: TensorScalar>(&self, name: &str, tensor: Tensor<T>) -> Result<(), Error> {
        let ExecutionContext(ctx) = self;
        ctx.set_input(name, tensor.into()).map_err(From::from)
    }

    /// Perform the inference, storing the computed output tensors.
    pub fn compute(&self) -> Result<(), Error> {
        let ExecutionContext(ctx) = self;
        ctx.compute().map_err(From::from)
    }

    /// Get the output tensor named `name`.
    ///
    /// The names of the output tensors depend on your model and
    /// should be found in its documentation.
    pub fn get_output<T: TensorScalar>(&self, name: &str) -> Result<Tensor<T>, Error> {
        let ExecutionContext(ctx) = self;
        let tensor = ctx.get_output(name)?;
        tensor.try_into()
    }

    /// Runs the model with the given inputs and returns the inferred output
    ///
    /// Takes as `inputs` an iterable over pairs of input tensor names
    /// and values (see [`ExecutionContext::set_input`]) and as
    /// `outputs` a list of output tensor names to retrieve (see
    /// [`ExecutionContext::get_output`]).
    ///
    /// Returns a HashMap from output tensor names to tensor values.
    /// The keys of the map are exactly the provided `outputs`.
    pub fn run<'os, T: TensorScalar>(
        &self,
        inputs: impl IntoIterator<Item = (impl AsRef<str>, Tensor<T>)>,
        outputs: &'os [impl AsRef<str> + 'os],
    ) -> Result<HashMap<&'os str, Tensor<T>>, Error> {
        for (input_name, input_tensor) in inputs {
            self.set_input(input_name.as_ref(), input_tensor)?;
        }
        self.compute()?;
        outputs
            .iter()
            .map(|output_name| Ok((output_name.as_ref(), self.get_output(output_name.as_ref())?)))
            .collect()
    }
}

/// A tensor
///
/// Used as input and output values of neural network graphs. Consists
/// of the actual tensor data, which is just a buffer over the scalar
/// type `T`, which can be any type implementing the trait
/// [`TensorScalar`], such as [`f32`] or [`u8`].
///
/// Also contains the dimensions of the tensor as a sequence of
/// positive integers.
///
#[derive(Debug)]
pub struct Tensor<T> {
    data: Vec<T>,
    dims: Vec<u32>,
}

impl<T> Tensor<T> {
    /// Construct a new tensor from a given data buffer and dimensions.
    ///
    /// `dims` must not be empty and the last element must be equal to
    /// 1, as the size of the scalar itself is encoded as
    /// `mem::size_of::<T>()`.
    pub fn new(data: Vec<T>, dims: Vec<u32>) -> Self {
        assert_eq!(dims.last(), Some(&1));
        Tensor { data, dims }
    }
}

impl<T: TensorScalar> From<Tensor<T>> for tensor::Tensor {
    fn from(mut value: Tensor<T>) -> Self {
        let data = Vec::from(unsafe {
            core::slice::from_raw_parts(
                value.data.as_mut_ptr().cast::<u8>(),
                value.data.len() * mem::size_of::<T>(),
            )
        });
        tensor::Tensor::new(&value.dims, tensor::TensorType::Fp32, &data)
    }
}

impl<T: TensorScalar> TryFrom<tensor::Tensor> for Tensor<T> {
    type Error = Error;

    fn try_from(value: tensor::Tensor) -> Result<Self, Self::Error> {
        if T::matches(value.ty()) {
            let mut data = value.data();
            Ok(Tensor {
                data: Vec::from(unsafe {
                    core::slice::from_raw_parts(
                        data.as_mut_ptr().cast::<T>(),
                        data.len() / mem::size_of::<T>(),
                    )
                }),
                dims: value.dimensions(),
            })
        } else {
            Err(Error::InvalidTensorType)
        }
    }
}

/// Any type that can be used as a scalar in a [`Tensor`].
///
/// # Safety
/// This trait is *unsafe* and you should not implement it yourself.
/// It is already implemented for exactly those types supported by
/// wasi-nn.
pub unsafe trait TensorScalar: Sized + Clone + Copy {
    fn tensor_type() -> tensor::TensorType;
    fn matches(t: tensor::TensorType) -> bool {
        Self::tensor_type() == t
    }
}

macro_rules! gen_tensor_scalar_impl {
    ($t:ty, $val:expr) => {
        unsafe impl TensorScalar for $t {
            fn tensor_type() -> tensor::TensorType {
                $val
            }
        }
    };
}

gen_tensor_scalar_impl!(u8, tensor::TensorType::U8);
gen_tensor_scalar_impl!(i32, tensor::TensorType::I32);
gen_tensor_scalar_impl!(i64, tensor::TensorType::I64);
gen_tensor_scalar_impl!(f32, tensor::TensorType::Fp32);
gen_tensor_scalar_impl!(f64, tensor::TensorType::Fp64);

impl<'a, T: TensorScalar + 'a, const BATCHSIZE: usize, const BATCHAMOUNT: usize>
    TryFrom<&'a Tensor<T>> for &'a [[T; BATCHSIZE]; BATCHAMOUNT]
{
    type Error = Error;

    /// Create two dimensional static array from tensor, fail if dims don't match N and M
    fn try_from(value: &'a Tensor<T>) -> Result<Self, Self::Error> {
        if value.dims[2] != 1
            || value.dims[1] as usize != BATCHSIZE
            || value.dims[0] as usize != BATCHAMOUNT
        {
            return Err(Error::InvalidTensorType);
        }

        let (chunks, &[]) = value.data.as_chunks::<BATCHSIZE>() else {
            return Err(Error::Bug("Model returned invalid output"));
        };

        chunks
            .try_into()
            .map_err(|_| Error::Bug("Model returned invalid output"))
    }
}

/// Error type for tha various errors that can occur during setup and inference.
#[derive(Debug)]
pub enum Error {
    /// An error originating in the wasi-nn API provided by
    /// the WASM runtime
    WasiNNError(errors::Error),
    /// Returned when the requested tensor type doesn't match what was
    /// computed by the model.
    InvalidTensorType,
    /// An I/O error, e.g. when reading model files
    IOError(io::Error),
    /// An actual bug. We use error type instead of panicing, because
    /// the wasi-http API requires that we always orderly exit and
    /// clean up the request handler. In case of e.g. a panic, we
    /// don't even get to read the panic message, which makes
    /// debugging hard.
    Bug(&'static str),
}

impl From<errors::Error> for Error {
    fn from(value: errors::Error) -> Self {
        Self::WasiNNError(value)
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::IOError(value)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Error::*;
        match self {
            WasiNNError(err) => {
                use errors::ErrorCode::*;
                let variant_description = match err.code() {
                    InvalidArgument => "invalid argument",
                    InvalidEncoding => "invalid encoding",
                    Timeout => "timeout",
                    RuntimeError => "runtime error",
                    UnsupportedOperation => "unsupported operation",
                    TooLarge => "too large",
                    NotFound => "not found",
                    Security => "security",
                    Unknown => "unknown error",
                };

                write!(f, "wasi_nn: {}", variant_description)?;
                if !err.data().is_empty() {
                    write!(f, ": {}", err.data())?;
                }
                writeln!(f)?;
            }
            InvalidTensorType => {
                writeln!(f, "tensor types don't match")?;
            }
            IOError(err) => {
                writeln!(f, "io error: {err}")?;
            }
            Bug(str) => {
                writeln!(f, "bug: {str}")?;
            }
        }
        Ok(())
    }
}

impl From<Error> for wasi::http::types::ErrorCode {
    fn from(value: Error) -> Self {
        Self::InternalError(Some(value.to_string()))
    }
}
