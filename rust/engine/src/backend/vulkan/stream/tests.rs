//! Compile-time properties of the stream and the backend.

use super::VulkanStream;
use crate::backend::vulkan::backend::VulkanBackend;

fn assert_send<T: Send>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn stream_and_backend_can_cross_threads() {
    assert_send::<VulkanStream>();
    assert_send_sync::<VulkanBackend>();
    // the trait requires `Stream: Send` and `Backend: Send + Sync`; the boxed forms are what the decoder holds
    assert_send::<Box<dyn crate::backend::Stream>>();
    assert_send_sync::<std::sync::Arc<dyn crate::backend::Backend>>();
}
