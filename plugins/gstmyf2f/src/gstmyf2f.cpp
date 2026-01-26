#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <yaml-cpp/yaml.h>
#include <atomic>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <dlfcn.h>
#include <nvbufsurface.h>

#include "processor_api.h"

#define GST_CAT_DEFAULT gst_myf2f_debug
GST_DEBUG_CATEGORY_STATIC (GST_CAT_DEFAULT);

typedef struct _DispatcherContext {
  void* so_handle;        // dlopen handle
  ProcessorAPI api;       // function table
  void* processor_ctx;    // opaque instance created by api.init
} DispatcherContext;

typedef struct _GstMyF2F {
  GstBaseTransform parent;

  gchar* config_path;
  gchar* processing_lib_config_path;
  gchar* processing_lib_path;
  std::mutex config_mutex;

  int width;
  int height;
  ProcPixelFormat pixel_format;

  std::atomic<bool> printed_once;
  DispatcherContext* dispatcher;
} GstMyF2F;

typedef struct _GstMyF2FClass {
  GstBaseTransformClass parent_class;
} GstMyF2FClass;

G_DEFINE_TYPE (GstMyF2F, gst_myf2f, GST_TYPE_BASE_TRANSFORM);


// ---------- Properties ----------
enum {
  PROP_0 = 0,
  PROP_CONFIG_PATH,
  PROP_PROCESSING_LIB_PATH,
  PROP_PROCESSING_LIB_CONFIG_PATH,
  PROP_DEBUG_FIRST_RUN,
};

static void gst_myf2f_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
  auto* self = (GstMyF2F*)object;
  switch (prop_id) {
    case PROP_CONFIG_PATH: {
      const gchar* p = g_value_get_string(value);
      std::lock_guard<std::mutex> lock(self->config_mutex);
      self->config_path = p ? g_strdup(p) : nullptr;
      break;
    }
    case PROP_PROCESSING_LIB_PATH: {
      const gchar* p = g_value_get_string(value);
      std::lock_guard<std::mutex> lock(self->config_mutex);
      g_free(self->processing_lib_path);
      self->processing_lib_path = p ? g_strdup(p) : nullptr;
      break;
    }
    case PROP_PROCESSING_LIB_CONFIG_PATH: {
      const gchar* p = g_value_get_string(value);
      std::lock_guard<std::mutex> lock(self->config_mutex);
      g_free(self->processing_lib_config_path);
      self->processing_lib_config_path = p ? g_strdup(p) : nullptr;
      break;
    }
    case PROP_DEBUG_FIRST_RUN: {
      self->printed_once.store(g_value_get_boolean(value), std::memory_order_relaxed);
      break;
    }
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
  }
}

static void gst_myf2f_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
  auto* self = (GstMyF2F*)object;
  switch (prop_id) {
    case PROP_CONFIG_PATH:
      g_value_set_string(value, self->config_path);
      break;
    case PROP_PROCESSING_LIB_PATH:
      g_value_set_string(value, self->processing_lib_path);
      break;
    case PROP_PROCESSING_LIB_CONFIG_PATH:
      g_value_set_string(value, self->processing_lib_config_path);
      break;
    case PROP_DEBUG_FIRST_RUN:
      g_value_set_boolean(value, self->printed_once.load(std::memory_order_relaxed));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
  }
}

static void gst_myf2f_finalize(GObject* object) {
  auto* self = (GstMyF2F*)object;

  if (self->dispatcher) {
    DispatcherContext* d = self->dispatcher;
    if (d->api.destroy && d->processor_ctx)
      d->api.destroy(d->processor_ctx);
    if (d->so_handle)
      dlclose(d->so_handle);
    g_free(d);
    self->dispatcher = nullptr;
  }

  std::lock_guard<std::mutex> lock(self->config_mutex);
  g_free(self->config_path);
  self->config_path = nullptr;
  g_free(self->processing_lib_path);
  self->processing_lib_path = nullptr;
  g_free(self->processing_lib_config_path);
  self->processing_lib_config_path = nullptr;

  G_OBJECT_CLASS(gst_myf2f_parent_class)->finalize(object);
}

// ---------- Config parsing ----------
static void parse_config_file(GstMyF2F* self) {
  gchar* config_path_copy = nullptr;
  gchar* lib_config_path_copy = nullptr;
  gchar* lib_path_copy = nullptr;
  {
    std::lock_guard<std::mutex> lock(self->config_mutex);
    if (!self->config_path) {
      GST_INFO_OBJECT(self, "No config-path set; using defaults.");
      return;
    }
    config_path_copy = g_strdup(self->config_path);
  }

  std::ifstream f(config_path_copy);
  if (!f.good()) {
    GST_WARNING_OBJECT(self, "Cannot open config file: %s", config_path_copy);
    g_free(config_path_copy);
    return;
  }

  // Use yaml-cpp for YAML parsing
  std::unordered_map<std::string, std::string> config_map;
  try {
    YAML::Node config = YAML::Load(f);
    for (YAML::const_iterator it = config.begin(); it != config.end(); ++it) {
      std::string key = it->first.as<std::string>();
      std::string value = it->second.as<std::string>();
      config_map[key] = value;
      g_print("Config: '%s' = '%s'\n", key.c_str(), value.c_str());
    }
  } catch (const std::exception& e) {
    GST_WARNING_OBJECT(self, "YAML parsing error: %s", e.what());
    g_free(config_path_copy);
    return;
  }

  // Set properties based on config_map
  for (const auto& kv : config_map) {
    if (kv.first == "config-path") {
      g_object_set(G_OBJECT(self), "config-path", kv.second.c_str(), nullptr);
    } else if (kv.first == "processing-lib-path") {
      g_object_set(G_OBJECT(self), "processing-lib-path", kv.second.c_str(), nullptr);
    } else if (kv.first == "processing-lib-config-path") {
      g_object_set(G_OBJECT(self), "processing-lib-config-path", kv.second.c_str(), nullptr);
    } else if (kv.first == "debug-first-run") {
      gboolean val = (kv.second == "1" || kv.second == "true");
      g_object_set(G_OBJECT(self), "debug-first-run", val, nullptr);
    }
  }
  g_free(config_path_copy);
}

// ---------- Start/Stop ----------
static gboolean gst_myf2f_start(GstBaseTransform* base) {
  auto* self = (GstMyF2F*)base;
  self->printed_once.store(false, std::memory_order_relaxed);
  parse_config_file(self);

  gchar* lib_path_copy = nullptr;
  gchar* lib_cfg_path_copy = nullptr;
  {
    std::lock_guard<std::mutex> lock(self->config_mutex);
    if (self->processing_lib_path)
      lib_path_copy = g_strdup(self->processing_lib_path);
    if (self->processing_lib_config_path)
      lib_cfg_path_copy = g_strdup(self->processing_lib_config_path);
  }

  if (!lib_path_copy) {
    GST_ERROR_OBJECT(self, "processing-lib-path is not set (config or property).");
    g_free(lib_cfg_path_copy);
    return FALSE;
  }

  // Allocate dispatcher
  self->dispatcher = (DispatcherContext*)g_new0(DispatcherContext, 1);
  DispatcherContext* d = self->dispatcher;

  // 1) dlopen
  d->so_handle = dlopen(lib_path_copy, RTLD_LAZY);
  if (!d->so_handle) {
    GST_ERROR_OBJECT(self, "dlopen failed for '%s': %s", lib_path_copy, dlerror());
    g_free(lib_path_copy);
    g_free(lib_cfg_path_copy);
    g_free(d);
    self->dispatcher = nullptr;
    return FALSE;
  }

  // 2) dlsym proc_register
  auto register_fn = (ProcStatus (*)(ProcessorAPI*))
      dlsym(d->so_handle, "proc_register");
  if (!register_fn) {
    GST_ERROR_OBJECT(self, "dlsym(proc_register) failed for '%s': %s",
                     lib_path_copy, dlerror());
    dlclose(d->so_handle);
    g_free(lib_path_copy);
    g_free(lib_cfg_path_copy);
    g_free(d);
    self->dispatcher = nullptr;
    return FALSE;
  }

  // 3) Fill ProcessorAPI
  memset(&d->api, 0, sizeof(ProcessorAPI));

  ProcStatus st = register_fn(&d->api);
  if (st != PROC_STATUS_OK ||
      !d->api.init || !d->api.process || !d->api.destroy) {
    GST_ERROR_OBJECT(self, "proc_register failed or incomplete API for '%s'", lib_path_copy);
    dlclose(d->so_handle);
    g_free(lib_path_copy);
    g_free(lib_cfg_path_copy);
    g_free(d);
    self->dispatcher = nullptr;
    return FALSE;
  }

  st = d->api.init(lib_cfg_path_copy, &d->processor_ctx);
  g_free(lib_cfg_path_copy);
  g_free(lib_path_copy);

  if (st != PROC_STATUS_OK || !d->processor_ctx) {
    GST_ERROR_OBJECT(self, "Processing lib init failed.");
    dlclose(d->so_handle);
    g_free(d);
    self->dispatcher = nullptr;
    return FALSE;
  }

  GST_INFO_OBJECT(self, "Processing lib initialized successfully.");
  return TRUE;
}

static gboolean gst_myf2f_stop(GstBaseTransform* base) {
  auto* self = (GstMyF2F*)base;

  if (self->dispatcher) {
    DispatcherContext* d = self->dispatcher;

    if (d->api.destroy && d->processor_ctx) {
      d->api.destroy(d->processor_ctx);
      d->processor_ctx = nullptr;
    }

    if (d->so_handle) {
      dlclose(d->so_handle);
      d->so_handle = nullptr;
    }

    g_free(d);
    self->dispatcher = nullptr;
  }

  return TRUE;
}

// ---------- Caps ----------
static gboolean gst_myf2f_set_caps(GstBaseTransform* base, GstCaps* incaps, GstCaps* outcaps) {
      GstMyF2F *self = (GstMyF2F*)base;

    if (!incaps) {
        GST_ERROR_OBJECT(self, "Incoming caps are NULL");
        return FALSE;
    }

    GstStructure *s = gst_caps_get_structure (incaps, 0);

    gst_structure_get_int(s, "width",  &self->width);
    gst_structure_get_int(s, "height", &self->height);

    const gchar *fmt = gst_structure_get_string(s, "format");
    if (!fmt) return FALSE;
    if (g_str_equal(fmt, "NV12") || g_str_equal(fmt, "I420")) {
        self->pixel_format = PROC_PIXFMT_NV12;
    } else if (g_str_equal(fmt, "RGB")) {
        self->pixel_format = PROC_PIXFMT_RGB;
    } else {
        GST_ERROR_OBJECT(self, "Unsupported pixel format: %s", fmt);
        return FALSE;
    }

    GST_INFO_OBJECT(self, "Negotiated caps: %dx%d format=%s",
                    self->width, self->height, fmt);

    g_print("[gstmyf2f] set_caps: width=%d height=%d format=%s\n",
            self->width, self->height, fmt);

    return TRUE;
}

// ---------- Transform (in-place) ----------
static GstFlowReturn gst_myf2f_transform_ip(GstBaseTransform* base, GstBuffer* buf) {
  auto* self = (GstMyF2F*)base;

  if (!self->dispatcher || !self->dispatcher->processor_ctx) {
    GST_ERROR_OBJECT(self, "Dispatcher or processor context not initialized.");
    return GST_FLOW_ERROR;
  }

  // Debug print once
  if (!self->printed_once.exchange(true, std::memory_order_acq_rel)) {
    // printf("[gstmyf2f] transform_ip: dispatching to processing lib (in-place mode)\n");
  }

  static int frame_count = 0;
  if (++frame_count % 30 == 0) {
    // printf("[gstmyf2f] Processed %d frames\n", frame_count);
  }

  // Map buffer (READ/WRITE)
  GstMapInfo map;
  if (!gst_buffer_map(buf, &map, GST_MAP_READWRITE)) {
    GST_ERROR_OBJECT(self, "Failed to map buffer.");
    return GST_FLOW_ERROR;
  }

  // Map NVMM surface
  NvBufSurface *surface = (NvBufSurface*)map.data;

  // Map surface for CPU access (read/write)
  int map_status = NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ_WRITE);
  if (map_status != 0) {
    gst_buffer_unmap(buf, &map);
    GST_ERROR_OBJECT(self, "Failed to map NVMM surface: %d", map_status);
    return GST_FLOW_ERROR;
  }

  // Sync surface for CPU access
  NvBufSurfaceSyncForCpu(surface, -1, -1);

  // Setup frame for processing library
  VP_Frame frame;
  frame.width  = self->width;
  frame.height = self->height;
  frame.stride = self->width;
  frame.data   = surface->surfaceList[0].mappedAddr.addr[0];
  frame.pixfmt = self->pixel_format;

  // Call processing library (processes in-place)
  ProcStatus st = self->dispatcher->api.process(self->dispatcher->processor_ctx, &frame);

  // Sync back to device before unmapping
  NvBufSurfaceSyncForDevice(surface, -1, -1);

  // Unmap surface
  NvBufSurfaceUnMap(surface, -1, -1);
  gst_buffer_unmap(buf, &map);

  return GST_FLOW_OK;
}

// ---------- Class/Init ----------
static void gst_myf2f_class_init(GstMyF2FClass* klass) {
  GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass* gstelement_class = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass* trans_class = GST_BASE_TRANSFORM_CLASS(klass);

  gobject_class->set_property = gst_myf2f_set_property;
  gobject_class->get_property = gst_myf2f_get_property;
  gobject_class->finalize = gst_myf2f_finalize;

  g_object_class_install_property(
      gobject_class, PROP_CONFIG_PATH,
      g_param_spec_string("config-path",
                          "Config file path",
                          "Path to key=value config file",
                          nullptr,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(
      gobject_class, PROP_PROCESSING_LIB_PATH,
      g_param_spec_string("processing-lib-path",
                          "Processing library path",
                          "Path to processing library file or directory",
                          nullptr,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(
      gobject_class, PROP_PROCESSING_LIB_CONFIG_PATH,
      g_param_spec_string("processing-lib-config-path",
                          "Processing lib config file path",
                          "Path to processing library config file",
                          nullptr,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(
      gobject_class, PROP_DEBUG_FIRST_RUN,
      g_param_spec_boolean("debug-first-run",
                          "Debug first run indication",
                          "Indicates if first run debug message was printed",
                          FALSE,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata(
      gstelement_class,
      "My Frame2Frame (in-place) filter",
      "Filter/Video",
      "Minimal in-place frame processor with config parsing",
      "Your Name <you@example.com>");

  GstCaps* caps = gst_caps_from_string(
    "video/x-raw(memory:NVMM), format=(string){ NV12, I420, RGB }, "
    "width=(int)[ 16, 8192 ], height=(int)[ 16, 8192 ], "
    "framerate=(fraction)[ 0/1, 240/1 ]; "
    "video/x-raw, format=(string){ NV12, I420, RGB }, "
    "width=(int)[ 16, 8192 ], height=(int)[ 16, 8192 ], "
    "framerate=(fraction)[ 0/1, 240/1 ]");

  gst_element_class_add_pad_template(
      gstelement_class, gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, gst_caps_ref(caps)));
  gst_element_class_add_pad_template(
      gstelement_class, gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));

  trans_class->start        = gst_myf2f_start;
  trans_class->stop         = gst_myf2f_stop;
  trans_class->set_caps     = gst_myf2f_set_caps;
  trans_class->transform_ip = gst_myf2f_transform_ip;  // In-place mode
  trans_class->passthrough_on_same_caps = TRUE;
}

static void gst_myf2f_init(GstMyF2F* self) {
  self->config_path = nullptr;
  self->processing_lib_path = nullptr;
  self->processing_lib_config_path = nullptr;
  self->printed_once.store(false, std::memory_order_relaxed);
  self->dispatcher = nullptr;
}

// ---------- Plugin entry ----------
static gboolean plugin_init(GstPlugin* plugin) {
  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "myf2f", 0, "myf2f");
  return gst_element_register(plugin, "myf2f", GST_RANK_NONE, gst_myf2f_get_type());
}

#define PACKAGE "myf2f"
GST_PLUGIN_DEFINE(
  GST_VERSION_MAJOR,
  GST_VERSION_MINOR,
  myf2f,
  "My Frame2Frame in-place filter plugin",
  plugin_init,
  "1.0.0",
  "LGPL",
  "gstmyf2f",
  "https://example.com"
)