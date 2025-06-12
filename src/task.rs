//! Task supervisor for managing supervised async tasks.
//!
//! The `TaskSupervisor` is a specialized version of [`DynamicSupervisor`](crate::DynamicSupervisor) that makes it easy
//! to run async tasks (futures) under supervision. It wraps each task in a lightweight actor that can be monitored
//! and restarted according to the configured policy.
//!
//! ## Use Cases
//!
//! The `TaskSupervisor` is ideal for:
//! - Background jobs that need supervision
//! - Periodic tasks that should be restarted on failure
//! - Long-running async operations that need monitoring
//! - Any async work that should be part of your supervision tree
//!
//! ## Key Features
//!
//! 1. **Simple API**: Wrap any async task in supervision with minimal boilerplate
//! 2. **Full Supervision**: Tasks get all the benefits of actor supervision
//! 3. **Flexible Policies**: Control restart behavior via [`TaskOptions`]
//! 4. **Resource Control**: Inherit `max_children` and other limits from `DynamicSupervisor`
//!
//! ## Example
//!
//! ```rust
//! use ractor_supervisor::*;
//! use ractor::concurrency::Duration;
//! use tokio::time::sleep;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Configure the task supervisor
//!     let options = TaskSupervisorOptions {
//!         max_children: Some(10),
//!         max_restarts: 3,
//!         max_window: Duration::from_secs(10),
//!         reset_after: Some(Duration::from_secs(30)),
//!     };
//!
//!     // Spawn the supervisor
//!     let (sup_ref, _) = TaskSupervisor::spawn(
//!         "task-sup".into(),
//!         options
//!     ).await.unwrap();
//!
//!     // Define a task that might fail
//!     let task_id = TaskSupervisor::spawn_task(
//!         sup_ref.clone(),
//!         || async {
//!             // Simulate some work
//!             sleep(Duration::from_secs(1)).await;
//!             
//!             // Maybe fail sometimes...
//!             if rand::random() {
//!                 panic!("Random failure!");
//!             }
//!
//!             Ok(())
//!         },
//!         TaskOptions::new()
//!             .name("periodic-job".into())
//!             .restart_policy(Restart::Permanent)
//!     ).await.unwrap();
//!
//!     // Later, stop the task if needed
//!     TaskSupervisor::terminate_task(sup_ref, task_id).await.unwrap();
//!
//!     ()
//! }
//! ```
//!
//! ## Task Lifecycle
//!
//! 1. When you call [`TaskSupervisor::spawn_task`]:
//!    - Your async task is wrapped in a [`TaskActor`]
//!    - The actor is spawned under the supervisor
//!    - The task starts executing immediately
//!
//! 2. If the task completes normally:
//!    - The actor stops normally
//!    - If policy is [`Restart::Permanent`], it's restarted
//!    - If policy is [`Restart::Transient`] or [`Restart::Temporary`], it's not restarted
//!
//! 3. If the task panics:
//!    - The actor fails abnormally
//!    - If policy is [`Restart::Permanent`] or [`Restart::Transient`], it's restarted
//!    - If policy is [`Restart::Temporary`], it's not restarted
//!
//! 4. Restart behavior is controlled by:
//!    - The [`TaskOptions::restart_policy`]
//!    - The supervisor's meltdown settings
//!    - Any configured backoff delays

use ractor::concurrency::Duration;
use ractor::concurrency::JoinHandle;
use ractor::{Actor, ActorCell, ActorName, ActorProcessingErr, ActorRef, SpawnErr};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use uuid::Uuid;

use crate::core::ChildSpec;
use crate::{
    ChildBackoffFn, DynamicSupervisor, DynamicSupervisorMsg, DynamicSupervisorOptions, Restart,
    SpawnFn,
};

/// Actor that wraps and executes an async task.
pub struct TaskActor;

/// Messages that can be sent to a [`TaskActor`].
pub enum TaskActorMessage {
    /// Execute the wrapped task.
    Run { task: TaskFn },
}

/// The result of a task execution.
type TaskFuture = Pin<Box<dyn Future<Output = Result<(), ActorProcessingErr>> + Send>>;

/// A wrapped async task that can be executed by a [`TaskActor`].
#[derive(Clone)]
pub struct TaskFn(Arc<dyn Fn() -> TaskFuture + Send + Sync>);

impl TaskFn {
    /// Create a new task wrapper from an async function.
    pub fn new<F, Fut>(factory: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), ActorProcessingErr>> + Send + 'static,
    {
        TaskFn(Arc::new(move || Box::pin(factory())))
    }
}

#[cfg_attr(feature = "async-trait", ractor::async_trait)]
impl Actor for TaskActor {
    type Msg = TaskActorMessage;
    type State = TaskFn;
    type Arguments = TaskFn;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        task: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(task)
    }

    async fn post_start(
        &self,
        myself: ActorRef<Self::Msg>,
        task: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        (task.0)().await?;
        myself.stop(None);
        Ok(())
    }
}

pub type TaskSupervisorMsg = DynamicSupervisorMsg;
pub type TaskSupervisorOptions = DynamicSupervisorOptions;

pub struct TaskSupervisor;

/// Options for configuring a task to be supervised.
pub struct TaskOptions {
    pub name: ActorName,
    pub restart: Restart,
    pub backoff_fn: Option<ChildBackoffFn>,
    /// Per-task "reset" duration: if a task has not failed for the given period,
    /// its failure count is reset.
    pub reset_after: Option<Duration>,
}

impl Default for TaskOptions {
    fn default() -> Self {
        Self {
            name: Uuid::new_v4().to_string(),
            restart: Restart::Temporary,
            backoff_fn: None,
            reset_after: None,
        }
    }
}

impl TaskOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn restart_policy(mut self, restart: Restart) -> Self {
        self.restart = restart;
        self
    }

    pub fn backoff_fn(mut self, backoff_fn: ChildBackoffFn) -> Self {
        self.backoff_fn = Some(backoff_fn);
        self
    }

    /// Set the per-task reset duration.
    pub fn reset_after(mut self, duration: Duration) -> Self {
        self.reset_after = Some(duration);
        self
    }
}

impl TaskSupervisor {
    pub async fn spawn(
        name: ActorName,
        options: TaskSupervisorOptions,
    ) -> Result<(ActorRef<TaskSupervisorMsg>, JoinHandle<()>), SpawnErr> {
        DynamicSupervisor::spawn(name, options).await
    }

    pub async fn spawn_linked(
        name: ActorName,
        startup_args: TaskSupervisorOptions,
        supervisor: ActorCell,
    ) -> Result<(ActorRef<TaskSupervisorMsg>, JoinHandle<()>), SpawnErr> {
        Actor::spawn_linked(Some(name), DynamicSupervisor, startup_args, supervisor).await
    }

    pub async fn spawn_task<F, Fut>(
        supervisor: ActorRef<TaskSupervisorMsg>,
        task: F,
        options: TaskOptions,
    ) -> Result<String, ActorProcessingErr>
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), ActorProcessingErr>> + Send + 'static,
    {
        let child_id = options.name;
        let task_wrapper = TaskFn::new(task);

        let spec = ChildSpec {
            id: child_id.clone(),
            spawn_fn: SpawnFn::new({
                let task_wrapper = task_wrapper.clone();
                move |sup, id| spawn_task_actor(id, task_wrapper.clone(), sup)
            }),
            restart: options.restart,
            backoff_fn: options.backoff_fn,
            reset_after: options.reset_after,
        };

        DynamicSupervisor::spawn_child(supervisor, spec).await?;
        Ok(child_id)
    }

    pub async fn terminate_task(
        supervisor: ActorRef<TaskSupervisorMsg>,
        task_id: String,
    ) -> Result<(), ActorProcessingErr> {
        DynamicSupervisor::terminate_child(supervisor, task_id).await
    }
}

async fn spawn_task_actor(id: String, task: TaskFn, sup: ActorCell) -> Result<ActorCell, SpawnErr> {
    let (child_ref, _join) = DynamicSupervisor::spawn_linked(id, TaskActor, task, sup).await?;
    Ok(child_ref.get_cell())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ractor::{
        call,
        concurrency::{sleep, Duration},
        ActorStatus,
    };
    use serial_test::serial;
    use tokio::sync::mpsc;

    async fn before_each() {
        sleep(Duration::from_millis(10)).await;
    }

    #[ractor::concurrency::test]
    #[serial]
    async fn test_basic_task_execution() {
        before_each().await;

        let (supervisor, handle) = TaskSupervisor::spawn(
            "test-supervisor".into(),
            TaskSupervisorOptions {
                max_children: Some(10),
                max_restarts: 3,
                max_window: Duration::from_secs(10),
                reset_after: Some(Duration::from_secs(30)),
            },
        )
        .await
        .unwrap();

        let (tx, mut rx) = mpsc::channel(1);

        let task_id = TaskSupervisor::spawn_task(
            supervisor.clone(),
            move || {
                let tx = tx.clone();
                async move {
                    tx.send(()).await.unwrap();
                    Ok(())
                }
            },
            TaskOptions::new().name("background-task".into()),
        )
        .await
        .unwrap();

        rx.recv().await.expect("Task should have executed");
        sleep(Duration::from_millis(100)).await;
        let state = call!(supervisor, DynamicSupervisorMsg::InspectState).unwrap();

        assert!(!state.active_children.contains_key(&task_id));

        supervisor.stop(None);
        let _ = handle.await;
    }

    #[ractor::concurrency::test]
    #[serial]
    async fn test_task_termination() {
        before_each().await;

        let (supervisor, handle) = TaskSupervisor::spawn(
            "test-supervisor".into(),
            TaskSupervisorOptions {
                max_children: Some(10),
                max_restarts: 3,
                max_window: Duration::from_secs(1),
                reset_after: Some(Duration::from_secs(1000)),
            },
        )
        .await
        .unwrap();

        let (tx, mut rx) = mpsc::channel(1);
        let task_id = TaskSupervisor::spawn_task(
            supervisor.clone(),
            move || {
                let tx = tx.clone();
                async move {
                    sleep(Duration::from_secs(10)).await;
                    tx.send(()).await.unwrap();
                    Ok(())
                }
            },
            TaskOptions::new().restart_policy(Restart::Permanent),
        )
        .await
        .unwrap();

        // Terminate before completion
        TaskSupervisor::terminate_task(supervisor.clone(), task_id.clone())
            .await
            .unwrap();

        // Verify task didn't complete
        let result = tokio::time::timeout(Duration::from_millis(100), rx.recv()).await;
        assert!(result.is_err(), "Task should have been terminated");

        supervisor.stop(None);
        let _ = handle.await;
    }

    #[ractor::concurrency::test]
    #[serial]
    async fn test_restart_policy() {
        before_each().await;

        let (supervisor, handle) = TaskSupervisor::spawn(
            "test-supervisor".into(),
            TaskSupervisorOptions {
                max_children: Some(10),
                max_restarts: 3,
                max_window: Duration::from_secs(1),
                reset_after: Some(Duration::from_secs(1000)),
            },
        )
        .await
        .unwrap();

        let (tx, mut rx) = mpsc::channel(3);
        let _task_id = TaskSupervisor::spawn_task(
            supervisor.clone(),
            move || {
                let tx = tx.clone();
                async move {
                    tx.send(()).await.unwrap();
                    panic!("Simulated failure");
                }
            },
            TaskOptions::new()
                .restart_policy(Restart::Transient)
                .name("restart-test".into()),
        )
        .await
        .unwrap();

        // Verify multiple restarts
        for _ in 0..4 {
            rx.recv().await.expect("Task should have restarted");
        }

        // Should be terminated after 3 failures (+1 initial)
        sleep(Duration::from_millis(100)).await;
        assert!(!supervisor
            .get_children()
            .iter()
            .any(|cell| cell.get_status() == ActorStatus::Running));

        supervisor.stop(None);
        let _ = handle.await;
    }

    #[ractor::concurrency::test]
    #[serial]
    async fn test_max_children_limit() {
        before_each().await;

        let (supervisor, handle) = TaskSupervisor::spawn(
            "max-child-supervisor".into(),
            TaskSupervisorOptions {
                max_children: Some(1),
                max_restarts: 3,
                max_window: Duration::from_secs(10),
                reset_after: None,
            },
        )
        .await
        .unwrap();

        // First task runs for a bit so it's still active when we try spawning another
        TaskSupervisor::spawn_task(
            supervisor.clone(),
            || async {
                sleep(Duration::from_millis(500)).await;
                Ok(())
            },
            TaskOptions::new().name("first-task".into()),
        )
        .await
        .unwrap();

        // Second task triggers meltdown by sending a SpawnChild message without a reply
        let task_fn = TaskFn::new(|| async { Ok(()) });
        let spec = ChildSpec {
            id: "second-task".into(),
            spawn_fn: SpawnFn::new(move |sup, id| {
                let task_fn = task_fn.clone();
                spawn_task_actor(id, task_fn, sup)
            }),
            restart: Restart::Temporary,
            backoff_fn: None,
            reset_after: None,
        };
        supervisor
            .send_message(DynamicSupervisorMsg::SpawnChild { spec, reply: None })
            .unwrap();

        // Supervisor should stop due to meltdown
        let _ = handle.await;
        assert_eq!(supervisor.get_status(), ActorStatus::Stopped);
    }
}
