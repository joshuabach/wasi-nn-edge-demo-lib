//! Utility wrappers around WASI HTTP types
//!
//! Primarily, this contains a [`RequestHandler`] trait, that client
//! could should implement for HTTP handler functionality.

// Re-export of type from wassi-http binding. that are used in the interface of
// [RequestHandler] or are useful for implementing it.
use wasi::http::types::{
    ErrorCode, Fields, IncomingBody, IncomingRequest, OutgoingBody, OutgoingResponse,
};
use wasi::io::streams::{InputStream, OutputStream};

use crate::interface;

use std::io::Write;

/// A request handler to provide convenience for handling WASI HTTP requests.
///
/// Probably you will only need to implement
/// [`handle_data`](RequestHandler::handle_data), which works with
/// [`interface`] types, however, you can also override the default
/// implementation of
/// [`handle_request`](RequestHandler::handle_request).
///
/// From your component handler, you only need to call
/// [`handle_request`](RequestHandler::handle_request), which might look like this:
///
/// ```
/// # trait Guest { fn handle(_: IncomingRequest, _: ResponseOutparam); }
/// use wasi_nn_demo_lib::{http::*, interface};
/// use wasi::exports::http::incoming_handler::*;
/// use wasi::http::types::ErrorCode;
///
/// struct Handler;
/// impl Handler {
///     fn new() -> Self { Self }
/// }
/// impl RequestHandler for Handler {
///     fn handle_data(&mut self, input: interface::DataWindow) -> Result<interface::InferenceResult, ErrorCode> {
///         todo!()
///     }
/// }
///
/// struct Component;
/// impl Guest for Component {
///     fn handle(request: IncomingRequest, response_outparam: ResponseOutparam) {
///         let mut handler = Handler::new();
///         let response = handler.handle_request(request);
///         ResponseOutparam::set(response_outparam, response);
///     }
/// }
/// ```
pub trait RequestHandler {
    /// Primary entry point to the handler, "converting" a request into a response or error.
    fn handle_request(&mut self, request: IncomingRequest) -> Result<OutgoingResponse, ErrorCode> {
        let headers = Fields::new();

        respond_with_output_stream(headers, |mut out| {
            let (mut input, body) = get_input_stream(request)?;
            let mut fun = || {
                let input_data: interface::DataWindow = serde_json::from_reader(&mut input)
                    .map_err(|e| {
                        ErrorCode::InternalError(Some(format!(
                            "Error reading JSON from incoming stream: {e}"
                        )))
                    })?;

                let output_data = self.handle_data(input_data)?;

                // For some reason `serde_json::to_writer(&mut out,
                // &data)` blocks indefinetly (stream never becomes ready
                // to write), so we serialize to a string and write that.
                let str = serde_json::to_string(&output_data)
                    .expect("values of type DataBatch should always be serializable as JSON");
                write!(out, "{}", str).map_err(|e| {
                    ErrorCode::InternalError(Some(format!("Error writing to outgoing stream: {e}")))
                })
            };

            // We need to explicity drop the input and finish the body
            // even in the case of an error. Since rust has no real
            // finally, we run the main body of this function as a
            // closure, keeping the result until we did the cleanup,
            // only then returning it.
            //
            // TODO: There must be a better way to handle this.
            let res = fun();

            drop(input);
            IncomingBody::finish(body);

            res
        })
    }

    /// Produce an inference result from a series of data points.
    ///
    /// Implement this function to call your time series model.
    fn handle_data(
        &mut self,
        input: interface::DataWindow,
    ) -> Result<interface::InferenceResult, ErrorCode>;
}

fn get_input_stream(request: IncomingRequest) -> Result<(InputStream, IncomingBody), ErrorCode> {
    let body = request
        .consume()
        .map_err(|()| ErrorCode::InternalError(Some("Bug: Cannot get request body".to_owned())))?;

    let stream = body.stream().map_err(|()| {
        ErrorCode::InternalError(Some("Bug: Cannot get incoming stream".to_owned()))
    })?;

    Ok((stream, body))
}

fn respond_with_output_stream(
    headers: Fields,
    handle_stream: impl FnOnce(OutputStream) -> Result<(), ErrorCode>,
) -> Result<OutgoingResponse, ErrorCode> {
    let resp = OutgoingResponse::new(headers);
    let body = resp
        .body()
        .map_err(|()| ErrorCode::InternalError(Some("Bug: Cannot get response body".to_owned())))?;
    body.write()
        .map_err(|()| ErrorCode::InternalError(Some("Bug: Cannot get outgoing stream".to_owned())))
        .and_then(handle_stream)?;

    OutgoingBody::finish(body, None)?;
    Ok(resp)
}
