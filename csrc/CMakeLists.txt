set(LIBRARY_NAME torch_mcpu)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/OpenRegSharedAllocator.cpp"
    PROPERTIES COMPILE_DEFINITIONS "TORCH_MCPU_SHARED_USE_PREMAPPED=$<BOOL:${TORCH_MCPU_SHARED_USE_PREMAPPED}>"
)

# ATen's OpenMP parallel_for implementation is header-only and is enabled by
# the compiler's _OPENMP feature macro. Keep this local to the RNG kernel so
# enabling uniform_ parallelism does not change unrelated torch_mcpu operators.
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/uniform.cpp"
    PROPERTIES COMPILE_OPTIONS "-fopenmp"
)
target_link_options(${LIBRARY_NAME} PRIVATE "-fopenmp")

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_cpu_library openreg)
target_include_directories(${LIBRARY_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/third_party/openreg/include
)
target_compile_definitions(${LIBRARY_NAME} PUBLIC
    TORCH_MCPU_ENABLE_MEMORY_PROTECTION=$<BOOL:${TORCH_MCPU_ENABLE_MEMORY_PROTECTION}>
    TORCH_MCPU_ENABLE_CPU_FALLBACK=$<BOOL:${TORCH_MCPU_ENABLE_CPU_FALLBACK}>
    TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH=$<BOOL:${TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH}>
)

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
