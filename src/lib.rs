#![feature(slice_as_chunks)]

//! Utility library for implementing WASI components that serve time
//! series inference services.
//!
//! At its heart, client code will need to implement
//! [`http::RequestHandler`]. See its documentation for further
//! usage instructions.

pub mod http;
pub mod interface;
pub mod nn;
