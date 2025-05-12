//! HTTP interface definitions
//!
//! Type definititions for handling requests from the data stream
//! handler. These are (de-)serializable, typically to/from JSON.
//!
//! Input is usually received from [`DataWindow`], while the response of
//! the component should be produced as [`InferenceResult`].
//!
//! You probably simply want to implement the trait function
//! [`crate::http::RequestHandler::handle_data`] for your handler
//! instead of parsing the types in this module yourself.

use std::collections::HashMap;

use chrono::prelude::*;
use serde::{Deserialize, Serialize};

/// A value of a data point, that can be either a string or float.
///
/// It is always represented by two adjacent keys "dataType" and
/// "value" in a map, where "dataType" (either "Number" or "String")
/// is used to tell the type, while "value" contains the actual value.
/// Even though two keys are used to parse a value, it is only
/// represented by a single enum (tagged union) in Rust. Many other
/// keys, corresponding to other data, could be present in the map.
///
/// Probably you will not parse this directly, but as a part of a
/// [`DataPoint`] within a [`DataWindow`]. However, directly parsing it
/// from JSON could look like this:
///
/// ```
/// # use wasi_nn_demo_lib::interface::Value;
/// let value: Value = serde_json::from_str(r#"{
///     "dataType": "Number",
///     "value": 43.098,
///     "some_other_key": true
/// }"#).unwrap();
/// assert_eq!(value, Value::Number(43.098));
/// ```
///
/// When using this type as part of a struct, most likely you will
/// want to mark it as `#[serde(flatten)]`, see e.g. [`DataPoint`].
#[derive(PartialEq, Debug, Deserialize, Serialize)]
#[serde(tag = "dataType", content = "value")]
pub enum Value {
    Number(f32),
    String(String),
}

/// A data point represent one point in a time series.
///
/// It consists of the two keys for the [Value] and additionaly a
/// "quality" and a "timestamp", which must be UTC.
///
/// Probably you will not parse this directly, but as a part of a
/// [`DataWindow`]. However, directly parsing it from JSON could look
/// like this:
///
/// ```
/// # use wasi_nn_demo_lib::interface::*;
/// let data_point: DataPoint = serde_json::from_str(r#"{
///     "dataType": "String",
///     "quality": 0,
///     "timestamp": "2024-08-07T09:01:23.322Z",
///     "value": "markerValue"
/// }"#).unwrap();
/// assert_eq!(data_point, DataPoint {
///     quality: Some(0),
///     timestamp: Some(chrono::NaiveDate::from_ymd_opt(2024, 8, 7).unwrap()
///         .and_hms_milli_opt(9,1,23,322).unwrap()
///         .and_utc()),
///     value: Value::String("markerValue".to_owned())
/// })
/// ```
#[derive(PartialEq, Debug, Deserialize, Serialize)]
pub struct DataPoint {
    /// The quality of the measured / recorded value
    pub quality: Option<usize>,
    /// The moment in time at which the value was recorded
    pub timestamp: Option<DateTime<Utc>>,
    /// The actual value of this data point
    #[serde(flatten)]
    pub value: Value,
}

/// A data window represents a portion of a time series of data points.
///
/// Each data point is labeled by an object id, which are the keys
/// into the `data` map. As such they are not inherently in order, but
/// must be sorted by [`DataPoint::timestamp`].
///
/// The JSON representation of a data window might look like this:
///
/// ```json
/// {
///   "Input1": {
///     "dataType": "Number",
///     "quality": 1024,
///     "timestamp": "2024-12-03T15:35:18.372Z",
///     "value": 43.0981827
///   },
///  "Input3": {
///     "dataType": "Number",
///     "quality": 1024,
///     "timestamp": "2024-12-03T15:35:18.372Z",
///     "value": -0.1
///   },
///  ...
///  "objectId": {
///     "dataType": "String",
///     "quality": 0,
///     "timestamp": "2024-08-07T09:01:23.322Z",
///     "value": "markerValue"
///   },
///   "previousAlarms": {}
/// }
/// ```
#[derive(Debug, Deserialize)]
pub struct DataWindow {
    #[serde(flatten)]
    pub data: HashMap<String, DataPoint>,
    /// This field is present (without value) in sample input
    #[serde(rename = "previousAlarms")]
    #[allow(clippy::zero_sized_map_values)]
    pub previous_alarms: Option<HashMap<String, ()>>,
}

/// The result of the inference of the component
///
/// This is a proof of concept definition of this type, that should be
/// adapted to the concrete use case.
#[derive(Debug, Serialize)]
pub enum InferenceResult {
    /// Most notably, in the `PredictedValues` variant, this is simply a
    /// stream of [`DataPoint`]s that represent the future values the
    /// model predicted based on the input.
    ///
    /// It could look like this:
    ///
    /// ```json
    /// {
    ///   "PredictedValues": [
    ///     {
    ///       "dataType": "Number",
    ///       "quality": 1024,
    ///       "timestamp": "2024-12-03T15:35:18.372Z",
    ///       "value": 43.0981827
    ///     },
    ///     {
    ///       "dataType": "Number",
    ///       "quality": 1024,
    ///       "timestamp": "2024-12-03T15:35:18.372Z",
    ///       "value": -0.1
    ///     },
    ///     ...
    ///   ]
    /// }
    /// ```
    PredictedValues(Vec<DataPoint>),
    /// If the model is not a forecasting model, but a predictor,
    /// the response could also be a map of labels to confidences.
    ///
    /// This is not used by the prototype.
    Classifications(HashMap<String, f32>),
}
