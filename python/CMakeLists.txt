set(TILE_FWK_WHL_NAME "tile_fwk-0.1.0-py3-none-any.whl")
set(TILE_FWK_PY_DIR ${TILE_FWK_BIN_ROOT}/python)
set(TILE_FWK_PY_BASE ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(${TILE_FWK_WHL_NAME}
    COMMAND echo "[TileFwk] Build target ${TILE_FWK_WHL_NAME} begin."
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${TILE_FWK_PY_DIR}/tile_fwk
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TILE_FWK_PY_DIR}/tile_fwk
    COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_PY_BASE}/tile_fwk/*.py ${TILE_FWK_PY_DIR}/tile_fwk/
    COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_PY_BASE}/setup.py ${TILE_FWK_PY_DIR}/
    COMMAND ${CMAKE_COMMAND} -E chdir ${TILE_FWK_PY_DIR} ${TILE_FWK_PYTHON3_EXE} setup.py bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E rename ${TILE_FWK_PY_DIR}/dist/tile_fwk*.whl ${TILE_FWK_PY_DIR}/${TILE_FWK_WHL_NAME}
    COMMAND echo "[TileFwk] Build target ${TILE_FWK_WHL_NAME} end."
)

install(FILES
    ${TILE_FWK_PY_DIR}/${TILE_FWK_WHL_NAME}
    DESTINATION lib OPTIONAL
)
