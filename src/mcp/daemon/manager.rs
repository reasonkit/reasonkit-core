//! Daemon Process Manager
//!
//! Handles lifecycle management for the MCP daemon process.
//!
//! # Design
//!
//! Uses a safe subprocess approach instead of fork() to comply with
//! the project's `#![deny(unsafe_code)]` policy. The daemon is spawned
//! as a detached child process running in "serve-daemon" mode.

use crate::error::{Error, Result};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use tracing::{info, warn};

/// Daemon status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonStatus {
    /// Daemon is running
    Running { pid: u32, uptime_secs: u64 },
    /// Daemon is stopped
    Stopped,
    /// PID file exists but process is dead
    Stale,
}

/// Daemon process manager
pub struct DaemonManager {
    pid_file: PathBuf,
    socket_path: PathBuf,
    log_path: PathBuf,
}

impl DaemonManager {
    /// Create new daemon manager
    pub fn new() -> Result<Self> {
        let data_dir = Self::get_data_dir()?;

        Ok(Self {
            pid_file: data_dir.join("daemon.pid"),
            socket_path: Self::get_socket_path()?,
            log_path: data_dir.join("logs").join("mcp-daemon.log"),
        })
    }

    /// Get data directory (XDG compliant)
    fn get_data_dir() -> Result<PathBuf> {
        let data_dir = dirs::data_local_dir()
            .ok_or_else(|| Error::config("Failed to get data directory"))?
            .join("reasonkit")
            .join("mcp");

        std::fs::create_dir_all(&data_dir)?;
        Ok(data_dir)
    }

    /// Get socket path (platform-specific)
    #[cfg(unix)]
    pub fn get_socket_path() -> Result<PathBuf> {
        use std::env;

        // Try XDG_RUNTIME_DIR first (systemd standard)
        let runtime_dir = env::var("XDG_RUNTIME_DIR")
            .ok()
            .map(PathBuf::from)
            .or_else(dirs::runtime_dir)
            .unwrap_or_else(env::temp_dir);

        let socket_dir = runtime_dir.join("reasonkit");
        std::fs::create_dir_all(&socket_dir)?;

        Ok(socket_dir.join("mcp.sock"))
    }

    #[cfg(windows)]
    pub fn get_socket_path() -> Result<PathBuf> {
        // Windows uses named pipes, but return path for consistency
        Ok(PathBuf::from(r"\\.\pipe\reasonkit-mcp"))
    }

    /// Get named pipe name (Windows only)
    #[cfg(windows)]
    pub fn get_pipe_name() -> String {
        format!(r"\\.\pipe\reasonkit-mcp-{}", whoami::username())
    }

    /// Start the daemon
    pub async fn start(&self) -> Result<()> {
        // Check if already running
        if matches!(self.status().await, DaemonStatus::Running { .. }) {
            return Err(Error::daemon("Daemon already running"));
        }

        info!("Starting MCP daemon...");

        // Platform-specific daemon spawn
        #[cfg(unix)]
        self.daemonize_unix()?;

        #[cfg(windows)]
        self.spawn_detached_windows()?;

        // Wait for daemon to start (check PID file)
        self.wait_for_start().await?;

        info!("MCP daemon started successfully");
        Ok(())
    }

    /// Stop the daemon
    pub async fn stop(&self) -> Result<()> {
        let status = self.status().await;

        match status {
            DaemonStatus::Running { pid, .. } => {
                info!("Stopping MCP daemon (PID {})...", pid);

                // Send shutdown signal
                self.send_shutdown_signal(pid)?;

                // Wait for graceful shutdown
                self.wait_for_shutdown(Duration::from_secs(10)).await?;

                // Cleanup
                self.cleanup()?;

                info!("MCP daemon stopped successfully");
                Ok(())
            }
            DaemonStatus::Stale => {
                warn!("Cleaning up stale PID file");
                self.cleanup()?;
                Ok(())
            }
            DaemonStatus::Stopped => {
                warn!("Daemon is not running");
                Ok(())
            }
        }
    }

    /// Restart the daemon
    pub async fn restart(&self) -> Result<()> {
        info!("Restarting MCP daemon...");
        self.stop().await?;
        tokio::time::sleep(Duration::from_secs(1)).await;
        self.start().await
    }

    /// Get daemon status
    pub async fn status(&self) -> DaemonStatus {
        match self.read_pid() {
            Ok(pid) => {
                if self.process_exists(pid) {
                    let uptime = self.get_uptime().unwrap_or(0);
                    DaemonStatus::Running {
                        pid,
                        uptime_secs: uptime,
                    }
                } else {
                    DaemonStatus::Stale
                }
            }
            Err(_) => DaemonStatus::Stopped,
        }
    }

    /// Read PID from file
    fn read_pid(&self) -> Result<u32> {
        let content = std::fs::read_to_string(&self.pid_file)
            .map_err(|_| Error::daemon("PID file not found"))?;

        content
            .trim()
            .parse::<u32>()
            .map_err(|_| Error::daemon("Invalid PID in file"))
    }

    /// Write PID to file
    fn write_pid(&self, pid: u32) -> Result<()> {
        std::fs::write(&self.pid_file, pid.to_string())?;
        Ok(())
    }

    /// Check if process exists (safe implementation using /proc on Unix)
    #[cfg(unix)]
    fn process_exists(&self, pid: u32) -> bool {
        // Check /proc/{pid} directory exists (Linux)
        let proc_path = PathBuf::from(format!("/proc/{}", pid));
        if proc_path.exists() {
            return true;
        }

        // Fallback: try to read process info via ps command
        Command::new("ps")
            .args(["-p", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[cfg(windows)]
    fn process_exists(&self, pid: u32) -> bool {
        // Use tasklist command to check if process exists (safe, no unsafe)
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {}", pid)])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .map(|o| {
                let stdout = String::from_utf8_lossy(&o.stdout);
                stdout.contains(&pid.to_string())
            })
            .unwrap_or(false)
    }

    /// Get daemon uptime (seconds)
    fn get_uptime(&self) -> Result<u64> {
        let metadata = std::fs::metadata(&self.pid_file)?;
        let created = metadata.created()?;
        let elapsed = std::time::SystemTime::now()
            .duration_since(created)
            .unwrap_or_default();
        Ok(elapsed.as_secs())
    }

    /// Spawn daemon as detached subprocess (safe, no fork)
    ///
    /// This uses `std::process::Command` to spawn the daemon as a child process
    /// with redirected stdio, avoiding the need for unsafe fork() calls.
    #[cfg(unix)]
    fn daemonize_unix(&self) -> Result<()> {
        use std::os::unix::process::CommandExt;

        // Get current executable
        let exe = std::env::current_exe()?;

        // Create log directory if needed
        if let Some(parent) = self.log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Open log file for stdout/stderr
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;

        let log_stderr = log_file.try_clone()?;

        // Spawn daemon as detached process
        let child = Command::new(&exe)
            .arg("mcp")
            .arg("serve-daemon")
            .arg("--socket")
            .arg(&self.socket_path)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_stderr))
            .current_dir("/") // Don't lock any mount points
            .process_group(0) // New process group (detach from terminal)
            .spawn()
            .map_err(|e| Error::daemon(format!("Failed to spawn daemon: {}", e)))?;

        // Write PID file
        self.write_pid(child.id())?;

        info!("Daemon spawned with PID {}", child.id());
        Ok(())
    }

    /// Spawn detached process on Windows
    #[cfg(windows)]
    fn spawn_detached_windows(&self) -> Result<()> {
        use std::os::windows::process::CommandExt;

        // CREATE_NO_WINDOW = 0x08000000 (constant to avoid winapi dependency)
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        const DETACHED_PROCESS: u32 = 0x00000008;

        let exe = std::env::current_exe()?;

        // Create log directory if needed
        if let Some(parent) = self.log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Open log file for stdout/stderr
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;

        let log_stderr = log_file.try_clone()?;

        let child = Command::new(exe)
            .arg("mcp")
            .arg("serve-daemon")
            .arg("--socket")
            .arg(&self.socket_path)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_stderr))
            .creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS)
            .spawn()
            .map_err(|e| Error::daemon(format!("Failed to spawn daemon: {}", e)))?;

        // Write PID
        self.write_pid(child.id())?;

        info!("Daemon spawned with PID {}", child.id());
        Ok(())
    }

    /// Send shutdown signal to process (safe implementation using kill command)
    #[cfg(unix)]
    fn send_shutdown_signal(&self, pid: u32) -> Result<()> {
        // Use kill command instead of libc kill()
        let status = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| Error::daemon(format!("Failed to run kill command: {}", e)))?;

        if !status.success() {
            return Err(Error::daemon(format!(
                "kill -TERM {} failed with exit code: {:?}",
                pid,
                status.code()
            )));
        }

        Ok(())
    }

    #[cfg(windows)]
    fn send_shutdown_signal(&self, pid: u32) -> Result<()> {
        // Use taskkill command instead of Windows API
        let status = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T"]) // /T = terminate child processes
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| Error::daemon(format!("Failed to run taskkill: {}", e)))?;

        if !status.success() {
            // Try forceful termination
            let status = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/F", "/T"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map_err(|e| Error::daemon(format!("Failed to run taskkill /F: {}", e)))?;

            if !status.success() {
                return Err(Error::daemon(format!("taskkill failed for PID {}", pid)));
            }
        }

        Ok(())
    }

    /// Wait for daemon to start
    async fn wait_for_start(&self) -> Result<()> {
        for _ in 0..20 {
            // Check every 500ms, max 10s
            tokio::time::sleep(Duration::from_millis(500)).await;

            if matches!(self.status().await, DaemonStatus::Running { .. }) {
                return Ok(());
            }
        }

        Err(Error::daemon("Daemon failed to start within 10 seconds"))
    }

    /// Wait for daemon to shutdown
    async fn wait_for_shutdown(&self, timeout: Duration) -> Result<()> {
        let start = std::time::Instant::now();

        while start.elapsed() < timeout {
            if matches!(self.status().await, DaemonStatus::Stopped) {
                return Ok(());
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Err(Error::daemon("Daemon did not shut down gracefully"))
    }

    /// Cleanup PID file and socket
    fn cleanup(&self) -> Result<()> {
        // Remove PID file
        if self.pid_file.exists() {
            std::fs::remove_file(&self.pid_file).ok();
        }

        // Remove socket (Unix only)
        #[cfg(unix)]
        {
            if self.socket_path.exists() {
                std::fs::remove_file(&self.socket_path).ok();
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_data_dir() {
        let dir = DaemonManager::get_data_dir().unwrap();
        assert!(dir.to_str().unwrap().contains("reasonkit"));
    }

    #[test]
    fn test_get_socket_path() {
        let path = DaemonManager::get_socket_path().unwrap();
        assert!(
            path.to_str().unwrap().contains("reasonkit") || path.to_str().unwrap().contains("pipe")
        );
    }

    #[tokio::test]
    async fn test_daemon_status_stopped() {
        let manager = DaemonManager::new().unwrap();
        let status = manager.status().await;
        assert_eq!(status, DaemonStatus::Stopped);
    }
}
