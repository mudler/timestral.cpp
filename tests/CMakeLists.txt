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
ts_add_test(test_predict_companions)

ts_add_test(test_masks)
add_test(NAME test_masks_units COMMAND test_masks --unit)
add_test(NAME test_masks_209m COMMAND test_masks --real)
ts_add_test(test_covariates)
add_test(NAME test_covariates_units COMMAND test_covariates --unit)
add_test(NAME test_covariates_209m COMMAND test_covariates --real)
ts_add_test(test_predict_block)
add_test(NAME test_predict_block_units COMMAND test_predict_block --unit)
add_test(NAME test_predict_block_209m COMMAND test_predict_block --real)

ts_add_test(test_refine)
add_test(NAME test_refine_units COMMAND test_refine --unit)
add_test(NAME test_refine_209m COMMAND test_refine --real)

ts_add_test(test_forecast)
add_test(NAME test_forecast_units COMMAND test_forecast --unit)
add_test(NAME test_forecast_209m COMMAND test_forecast --real)
set_tests_properties(test_forecast_209m PROPERTIES LABELS "slow")

ts_add_test(test_f16)
ts_add_test(test_q8)

ts_add_test(test_linear)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_linear PRIVATE -ffp-contract=off)
endif()

ts_add_test(test_forecast_determinism)

ts_add_test(test_attention)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_attention PRIVATE -ffp-contract=off)
endif()

ts_add_test(test_capi)
target_sources(test_capi PRIVATE test_capi_c99.c)
set_source_files_properties(test_capi_c99.c PROPERTIES LANGUAGE C)
set_target_properties(test_capi PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
if(UNIX AND TIMESTRAL_SHARED)
  target_compile_definitions(test_capi PRIVATE TS_CAPI_SHARED_PATH="$<TARGET_FILE:timestral>")
  target_link_libraries(test_capi PRIVATE ${CMAKE_DL_LIBS})
endif()
add_test(NAME test_capi_209m COMMAND test_capi --real)
set_tests_properties(test_capi_209m PROPERTIES SKIP_RETURN_CODE 77)

# CLI transport harness needs ctypes (shared library), but no numerical Python
# dependency for the tiny checks. Real fixture loading additionally uses gguf.
if(TIMESTRAL_SHARED AND TIMESTRAL_BUILD_CLI)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME test_cli COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cli.py
      --cli $<TARGET_FILE:timestral-cli> --library $<TARGET_FILE:timestral>
      --model ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny/timestral-tiny-f32.gguf
      --scratch ${CMAKE_CURRENT_BINARY_DIR}/cli-fixtures)
  endif()
endif()


ts_add_test(test_forecast_anchors)
add_test(NAME test_forecast_anchors_209m COMMAND test_forecast_anchors --real)
set_tests_properties(test_forecast_anchors_209m PROPERTIES SKIP_RETURN_CODE 77)

if(TIMESTRAL_BUILD_EVAL_DRIVER)
  ts_add_test(test_eval_driver)
  target_link_libraries(test_eval_driver PRIVATE timestral-eval-codec)
  add_test(NAME test_eval_driver_209m COMMAND test_eval_driver --real)
  set_tests_properties(test_eval_driver_209m PROPERTIES SKIP_RETURN_CODE 77)
endif()
