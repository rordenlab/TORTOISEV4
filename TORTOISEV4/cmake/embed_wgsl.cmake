# Wrap a .wgsl file in a C++ raw string literal so shaders are compiled into the
# binary - no runtime asset paths to get wrong.
#   cmake -DIN=x.wgsl -DOUT=x.wgsl.h -DVAR=kX -P embed_wgsl.cmake
file(READ ${IN} _src)
# WGSL has no #include; metric shaders share a prelude, prepended here.
get_filename_component(_dir ${IN} DIRECTORY)
get_filename_component(_base ${IN} NAME)
if(_base MATCHES "^metric_" AND NOT _base STREQUAL "metric_common.wgsl")
    file(READ ${_dir}/metric_common.wgsl _prelude)
    set(_src "${_prelude}\n${_src}")
endif()
if(_src MATCHES "\\)WGSL")
    message(FATAL_ERROR "${IN} contains the raw-string terminator )WGSL")
endif()
get_filename_component(_name ${IN} NAME)
file(WRITE ${OUT}
     "// Generated from ${_name} - do not edit.\n"
     "#pragma once\n"
     "static const char *${VAR} = R\"WGSL(\n${_src})WGSL\";\n")
