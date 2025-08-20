set(TILE_FWK_WHL_NAME "tile_fwk-0.1.0-py3-none-any.whl")
set(TILE_FWK_PY_DIR ${TILE_FWK_BIN_ROOT}/python)
set(TILE_FWK_PY_BASE ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT DEFINED HI_PYTHON)
    set(HI_PYTHON python3)
endif()

add_custom_target(${TILE_FWK_WHL_NAME}
    COMMAND echo "[TileFwk] Build target ${TILE_FWK_WHL_NAME} begin." &&
    mkdir -p ${TILE_FWK_PY_DIR}/tile_fwk && rm -rf ${TILE_FWK_PY_DIR}/tile_fwk/* &&
    cp -rf ${TILE_FWK_PY_BASE}/tile_fwk/* ${TILE_FWK_PY_DIR}/tile_fwk/ &&
    cp -rf ${TILE_FWK_PY_BASE}/setup.py ${TILE_FWK_PY_DIR}/ &&
    cd ${TILE_FWK_PY_DIR} && ${HI_PYTHON} setup.py bdist_wheel &&
    cd - && ls && pwd &&
    ${CMAKE_COMMAND} -E rename ${TILE_FWK_PY_DIR}/dist/tile_fwk*.whl ${TILE_FWK_PY_DIR}/${TILE_FWK_WHL_NAME} &&
    echo "[TileFwk] Build target ${TILE_FWK_WHL_NAME} end."
)

install(FILES
    ${TILE_FWK_PY_DIR}/${TILE_FWK_WHL_NAME}
    DESTINATION lib OPTIONAL
)
