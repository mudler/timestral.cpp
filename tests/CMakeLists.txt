function(ts_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE timestral)
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${name} PRIVATE TS_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}"
                                             TS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

ts_add_test(test_smoke)
ts_add_test(test_parity_helpers)
ts_add_test(test_model_loader)
# The same executable on the real 209M file: skipped (77) when TIMESTRAL_TEST_GGUF is unset.
add_test(NAME test_model_loader_209m COMMAND test_model_loader --real)
set_tests_properties(test_model_loader_209m PROPERTIES SKIP_RETURN_CODE 77)
