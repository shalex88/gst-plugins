#ifndef PROCESSOR_API_H
#define PROCESSOR_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>  /* for uint32_t, uint8_t */

typedef enum {
    PROC_STATUS_OK          = 0,
    PROC_STATUS_ERR_GENERAL = -1,
    PROC_STATUS_ERR_CONFIG  = -2,
    PROC_STATUS_ERR_ALLOC   = -3,
    PROC_STATUS_ERR_RUNTIME = -4,
    PROC_STATUS_ERR_UNSUPPORTED = -5,
    PROC_STATUS_ERR_NOMEM   = -6
} ProcStatus;

typedef enum {
    PROC_PIXFMT_UNKNOWN = 0,
    PROC_PIXFMT_NV12,
    PROC_PIXFMT_RGB
} ProcPixelFormat;

typedef struct {
    int width;
    int height;
    int stride;              /* bytes per row */
    ProcPixelFormat pixfmt;
    void* data;
} VP_Frame;

typedef struct ProcessorAPI {
    /* Create a new processor instance. The implementation allocates *ctx. */
    ProcStatus (*init)(const char* config_path, void** ctx);

    /* Process one frame. ctx is the instance state.
     * For in-place processing, pass the same frame as input.
     */
    ProcStatus (*process)(void* ctx,
                          VP_Frame* frame);

    /* Destroy a processor instance created by init(). */
    void (*destroy)(void* ctx);
} ProcessorAPI;

ProcStatus proc_register(ProcessorAPI* api);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROCESSOR_API_H */