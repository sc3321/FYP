
find_package(Threads REQUIRED)

set(TARGET phaseGuard)


add_library(phaseGuard STATIC
    src/phaseLib/gpuPhase.cpp
    src/phaseLib/eventHandler.cpp
    src/policy/policyManager.cpp
    src/policy/memManager.cpp
)

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(${TARGET} PUBLIC cxx_std_17)

target_link_libraries(phaseGuard PUBLIC
    Threads::Threads
)

if (UNIX AND NOT APPLE)
    target_link_libraries(phaseGuard PUBLIC rt)
endif()
