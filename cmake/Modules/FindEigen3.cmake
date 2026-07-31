find_path(EIGEN3_INCLUDE_DIR
  NAMES Eigen/Dense
  PATHS
    ${CMAKE_SOURCE_DIR}/src/Eigen
    /mingw64/include/eigen3
    /usr/include/eigen3
    /usr/local/include/eigen3
    /opt/homebrew/include/eigen3
    "D:/msys64/mingw64/include/eigen3"
    "C:/msys64/mingw64/include/eigen3"
    $ENV{EIGEN3_INCLUDE_DIR}
)

find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIR)

mark_as_advanced(EIGEN3_INCLUDE_DIR)

if(Eigen3_FOUND AND NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
  )
endif()
