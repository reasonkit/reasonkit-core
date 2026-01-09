//! Signal Handling
//!
//! Sets up Unix signal handlers for graceful shutdown.

use crate::error::Result;
use tokio::sync::broadcast;
use tracing::info;

/// Setup signal handlers for graceful shutdown
#[cfg(unix)]
pub fn setup_signal_handlers() -> Result<broadcast::Sender<()>> {
    use tokio::signal::unix::{signal, SignalKind};

    let (shutdown_tx, _) = broadcast::channel(1);
    let shutdown_tx_clone = shutdown_tx.clone();

    tokio::spawn(async move {
        let mut sigterm = signal(SignalKind::terminate()).expect("Failed to setup SIGTERM handler");
        let mut sigint = signal(SignalKind::interrupt()).expect("Failed to setup SIGINT handler");

        tokio::select! {
            _ = sigterm.recv() => {
                info!("Received SIGTERM, initiating shutdown");
                shutdown_tx_clone.send(()).ok();
            }
            _ = sigint.recv() => {
                info!("Received SIGINT, initiating shutdown");
                shutdown_tx_clone.send(()).ok();
            }
        }
    });

    Ok(shutdown_tx)
}

#[cfg(windows)]
pub fn setup_signal_handlers() -> Result<broadcast::Sender<()>> {
    use tokio::signal::windows;

    let (shutdown_tx, _) = broadcast::channel(1);
    let shutdown_tx_clone = shutdown_tx.clone();

    tokio::spawn(async move {
        let mut ctrl_c = windows::ctrl_c().expect("Failed to setup Ctrl-C handler");
        let mut ctrl_break = windows::ctrl_break().expect("Failed to setup Ctrl-Break handler");

        tokio::select! {
            _ = ctrl_c.recv() => {
                info!("Received Ctrl-C, initiating shutdown");
                shutdown_tx_clone.send(()).ok();
            }
            _ = ctrl_break.recv() => {
                info!("Received Ctrl-Break, initiating shutdown");
                shutdown_tx_clone.send(()).ok();
            }
        }
    });

    Ok(shutdown_tx)
}
