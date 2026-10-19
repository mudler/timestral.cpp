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

ts_add_test(test_scaler)
add_test(NAME test_scaler_units COMMAND test_scaler --unit)
add_test(NAME test_scaler_209m COMMAND test_scaler --real)
set_tests_properties(test_scaler_209m PROPERTIES SKIP_RETURN_CODE 77)

ts_add_test(test_patch_embed)
ts_add_test(test_rope)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_rope PRIVATE -ffp-contract=off)
endif()
add_test(NAME test_patch_embed_units COMMAND test_patch_embed --unit)
add_test(NAME test_rope_units COMMAND test_rope --unit)
# Real component parity is opt-in while reference artifacts are generated.
# Invoke the executables with --real and TIMESTRAL_TEST_REAL_CASES=all or a
# comma-separated explicit subset; subset output cannot claim full acceptance.
ts_add_test(test_layer_time)
add_test(NAME test_layer_time_units COMMAND test_layer_time --unit)
add_test(NAME test_layer_time_209m COMMAND test_layer_time --real)
set_tests_properties(test_layer_time_209m PROPERTIES SKIP_RETURN_CODE 77)
ts_add_test(test_layer_variate)
ts_add_test(test_layer_time_variate)
ts_add_test(test_layer_variate_time)
add_test(NAME test_layer_time_variate_209m COMMAND test_layer_time_variate --real)
set_tests_properties(test_layer_time_variate_209m PROPERTIES SKIP_RETURN_CODE 77)

foreach(kind variate time_variate variate_time)
  add_test(NAME test_layer_${kind}_units COMMAND test_layer_${kind} --unit)
endforeach()

ts_add_test(test_encoder)
add_test(NAME test_encoder_units COMMAND test_encoder --unit)
# Strict real gate: sparse canonical dumps cannot satisfy every-layer acceptance.
add_test(NAME test_encoder_209m COMMAND test_encoder --real)
set_tests_properties(test_encoder_209m PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME test_encoder_negative_layer6_209m COMMAND test_encoder --real --negative-layer6)
set_tests_properties(test_encoder_negative_layer6_209m PROPERTIES
  SKIP_RETURN_CODE 77 ENVIRONMENT "TIMESTRAL_TEST_REAL_CASES=uni_short")

ts_add_test(test_head)
add_test(NAME test_head_units COMMAND test_head --unit)
add_test(NAME test_head_209m COMMAND test_head --real)
ts_add_test(test_predict_metric)
