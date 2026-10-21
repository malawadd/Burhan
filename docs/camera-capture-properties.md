# Camera capture negotiation and reported properties

`LocalSource` now makes a best-effort `CAP_PROP_FPS` request using the configured input FPS, alongside width, height and buffer size requests. Windows still selects the verified Media Foundation backend. This change does not establish a new measured camera throughput result.

The runtime description and camera probe expose `capture_properties`:

| Field | Meaning |
|---|---|
| `fps_requested` | FPS requested from the camera driver |
| `fps_request_accepted` | Boolean returned by the property setter; acceptance alone does not prove the device delivers that rate |
| `driver_fps` | Positive finite value returned by `CAP_PROP_FPS`, or `null` if unavailable; this is not a measured arrival rate |
| `requested_dimensions` | Configured width and height |
| `width_request_accepted`, `height_request_accepted` | Driver setter return values |
| `driver_dimensions` | Dimensions reported by the driver after negotiation |
| `effective_dimensions` | Initially driver dimensions; replaced with actual decoded frame dimensions after the first frame, and updated if frame dimensions change |
| `dimensions_source` | `driver_reported` or `observed_frame` |
| `buffer_size_requested`, `buffer_request_accepted` | Request for one buffered frame and driver setter result |
| `driver_buffer_size` | Positive finite driver buffer property, otherwise `null` |
| `driver_fps_is_measured` | Always `false`; protects against interpreting the driver declaration as a benchmark |
| `exposure_latency_ms` | `null`; exposure and sensor/backend delay remain unknown |
| `property_errors` | Unsupported property exceptions, when supplied by the backend |

Requested FPS can differ from driver-reported FPS and from achieved capture/inspection throughput. The runtime separately reports observed host capture and processed frame rates. These include their stated timing scopes; they do not measure sensor exposure latency. Buffer acceptance does not establish a low-latency camera path or production item coverage.

Metadata uses copy-on-write updates when observed dimensions become available. The runtime copies source metadata into its description and preserves it across reset/calibrate actions. Source stubs without this optional property use an empty object, retaining existing lightweight test compatibility.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest tests/test_camera_properties.py -q
```

Four mocked-camera tests verify that FPS is actually requested, rejected setters remain visible, driver declarations differ from observed dimensions, unsupported/nonfinite properties do not prevent capture, earlier metadata snapshots remain unchanged, and runtime reset preserves the negotiated properties even when the first frame publishes its actual dimensions during profile refresh. No real camera was opened for these tests.

The focused integration run also included the existing stub background-profile lifecycle test: **5 passed, 11 deselected**, with one existing Starlette/AnyIO deprecation warning. No heavy model inference ran.
