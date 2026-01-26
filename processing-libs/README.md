# Processor API
## Overview

The **Processor API** defines the binary interface (ABI) used by the **F2F GStreamer plugin** (`gstmyf2f`) to dynamically load and communicate with external **video-processing modules**.

This API enables a **plugin-of-a-plugin architecture**:

* The GStreamer plugin is the *host*
* The processing modules (`.so` libraries) are *plugins*
* Modules implement a unified interface for initialization, per-frame processing, and destruction
* The host discovers modules at runtime via `dlopen()` + `dlsym()`

This provides:

* Full modularity
* Hot-swapping of processing engines
* Multiple independent processor types (stabilization, super-res, denoise…)
* Clean ABI boundaries
* No global state
* No compile-time coupling

---

## Processor API Role

* `processor_api.h` defines the **one and only contract** that every video-processing module must obey.
* Every module lives in its own repository and includes this header.
* The GStreamer plugin also includes this header to interact with the modules.

This file is the **shared language** of the video-processing ecosystem.

## High-Level Architecture
```
Application User
      │
      ▼
GStreamer Pipeline
      │
      ▼
gstmyf2f plugin (host)
      │
   dlopen()
      ▼
Processor Module (.so)
      │
 proc_register()
      ▼
ProcessorAPI { init, process, destroy }
```

The plugin acts as a **module host**, dynamically loading `.so` modules that implement the Processor API.

---

# Processor API Responsibilities

Every processor module must:

1. Export **one function**:

   ```c
   ProcStatus proc_register(ProcessorAPI* api);
   ```

2. Fill in a **function table** containing:

   * `init()`
   * `process()`
   * `destroy()`

3. Maintain its own private per-instance **opaque context (ctx)**.

4. Follow the ABI rules in `processor_api.h`.

5. Avoid using global/shared state so that multiple processors can run concurrently.

---

## Life Cycle of a Processor Instance

### 1. Plugin loads the module

```c
handle = dlopen("/path/to/libprocessor.so", RTLD_LAZY)
register_fn = dlsym(handle, "proc_register")
```

### 2. Plugin registers module API

```c
ProcessorAPI api = {0};
api.api_version = PROCESSOR_API_VERSION;
register_fn(&api);
```

### 3. Plugin initializes the processor

```c
api.init(&cfg, &ctx);
```

### 4. Plugin processes each frame

```c
api.process(ctx, &frame_in, &frame_out);
```

### 5. Plugin destroys processor

```c
api.destroy(ctx);
dlclose(handle);
```

The plugin **never** inspects or modifies `ctx`.

---

## Data Structures

The API provides:

**`VP_Frame`** — representing the current frame
* **`ProcPixelFormat`** — enumeration of supported pixel formats
* **`ProcStatus`** — standard return codes

All structures are:

* POD-only
* No STL
* No C++ constructs
* ABI stable

---

## Implementing a Processor Module

Your module must:

### 1. Include the header:

```c
#include "processor_api.h"
```

### 2. Define your internal context:

```c
typedef struct {
    int frame_count;
    float* history;   // e.g. stabilization history
} MyProcCtx;
```

### 3. Implement init/process/destroy:

```c
static ProcStatus my_init(const VP_Config* cfg, void** ctx);
static ProcStatus my_process(void* ctx, const VP_FrameIn* in, VP_FrameOut* out);
static void       my_destroy(void* ctx);
```

### 4. Export `proc_register()`:

```c
ProcStatus proc_register(ProcessorAPI* api)
{
    api->init = my_init;
    api->process = my_process;
    api->destroy = my_destroy;
    return PROC_STATUS_OK;
}
```

### 5. Compile as a shared library:

```
libvproc_stabilization.so
libvproc_superres.so
libvproc_denoise.so
...
```

---

## Design Principles

* **Opaque Handle Pattern**
  Plugin never sees module internals.

* **Zero ABI entanglement**
  Modules only expose one symbol.

* **Dynamic selection**
  User picks module via config file path.

* **Reentrant & multi-instance safe**
  No global state. All module state is per `ctx`.

---

## Why This Architecture?

This design enables:

* Hot-swapping processing engines
* Modular video processing pipeline
* Easy integration of new algorithms
* Deployment of experimental or proprietary modules
* Better code isolation
* Clear plugin boundaries
* Safe ABI across dynamic libraries
* Submodule-based repo structure

This is the **state-of-the-art** plugin-of-a-plugin model used in graphics, AI, video, audio, and game engines.

---