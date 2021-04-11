#include <stdio.h>
#include <stdarg.h>

#include "logger.h"

// @temporary to test that windows build is working
void log_printf(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);

  va_end(args);

}
